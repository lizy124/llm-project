# 06 LoadConfig 与 ModelLoader 选择

本篇梳理 `LoadConfig` 与 `model_loader` 的关系。`LoadConfig` 是配置链路中和权重文件加载最直接相关的子配置；真正加载模型时，`get_model_loader(load_config)` 会根据 `load_format` 选择具体 loader。

## 1. `LoadConfig` 的职责

定义：`code/vllm/vllm/config/load.py:25`。

后处理/校验位置：

- `code/vllm/vllm/config/load.py:134`
- `code/vllm/vllm/config/load.py:138`

核心字段语义：

| 字段 | 作用 |
|---|---|
| `load_format` | 决定用哪个 loader / iterator 读取权重。 |
| `download_dir` | HF / ModelScope 下载缓存目录。 |
| `model_loader_extra_config` | 传给特殊 loader 的额外配置。 |
| `ignore_patterns` | 权重文件过滤模式。 |
| safetensors 相关策略 | 控制 safetensors 加载和 fallback 行为。 |

`LoadConfig` 不负责判断模型架构，也不负责权重如何映射到层；它负责告诉加载系统“权重文件从哪里来、以什么格式读、哪些文件忽略”。

## 2. loader 选择入口

入口：`code/vllm/vllm/model_executor/model_loader/__init__.py:122`。

简化逻辑：

```text
get_model_loader(load_config)
  ↓
读取 load_config.load_format
  ↓
查表得到 loader class
  ↓
实例化 loader(load_config)
```

`get_model()` 是进一步包装：`code/vllm/vllm/model_executor/model_loader/__init__.py:130`。

```text
get_model(vllm_config)
  ↓
loader = get_model_loader(vllm_config.load_config)
  ↓
loader.load_model(vllm_config, model_config)
```

## 3. `load_format -> loader` 映射

当前主要映射：

| load_format | loader | 说明 |
|---|---|---|
| `auto` | `DefaultModelLoader` | 默认路径，自动识别常规权重格式。 |
| `hf` | `DefaultModelLoader` | HuggingFace 常规权重。 |
| `safetensors` | `DefaultModelLoader` | safetensors 权重。 |
| `pt` | `DefaultModelLoader` | PyTorch `.bin` / `.pt` 权重。 |
| `npcache` | `DefaultModelLoader` | numpy cache 权重路径。 |
| `fastsafetensors` | `DefaultModelLoader` | fastsafetensors iterator。 |
| `instanttensor` | `DefaultModelLoader` | instant tensor iterator。 |
| `mistral` | `DefaultModelLoader` | Mistral 权重格式主路径。 |
| `bitsandbytes` | `BitsAndBytesModelLoader` | BnB 量化权重。 |
| `dummy` | `DummyModelLoader` | 构造 dummy 权重，不读真实 checkpoint。 |
| `sharded_state` | `ShardedStateLoader` | rank-aware 预分片 checkpoint。 |
| `runai_streamer` | `RunaiModelStreamerLoader` | RunAI streaming 读取。 |
| `runai_streamer_sharded` | `ShardedStateLoader` | RunAI sharded state 路径。 |
| `tensorizer` | `TensorizerLoader` | tensorizer 序列化格式。 |
| `modelexpress` | `ModelExpressModelLoader` | 外部 ModelExpress 集成。 |

映射入口：`code/vllm/vllm/model_executor/model_loader/__init__.py:122`。

## 4. `BaseModelLoader` 生命周期模板

定义：`code/vllm/vllm/model_executor/model_loader/base_loader.py:37`。

`load_model()` 模板入口：`code/vllm/vllm/model_executor/model_loader/base_loader.py:42`。

通用生命周期：

```text
BaseModelLoader.load_model(vllm_config, model_config)
  ↓
设置 dtype / device 上下文
  ↓
initialize_model(vllm_config, prefix="")
  ↓
self.load_weights(model, model_config)
  ↓
如果存在 online quant，finalize_layerwise_processing(...)
  ↓
process_weights_after_loading(model, model_config, target_device)
  ↓
model.eval()
  ↓
返回 nn.Module
```

它体现了一个重要设计：

- `BaseModelLoader` 控制生命周期；
- 子类实现 `load_weights()`；
- 特殊 loader 可以重写 `load_model()`。

## 5. `DefaultModelLoader`

默认主路径入口：`code/vllm/vllm/model_executor/model_loader/default_loader.py:382`。

核心方法：

- `_prepare_weights(...)`：`code/vllm/vllm/model_executor/model_loader/default_loader.py:97`
- `_get_weights_iterator(...)`：`code/vllm/vllm/model_executor/model_loader/default_loader.py:211`
- `get_all_weights(...)`：`code/vllm/vllm/model_executor/model_loader/default_loader.py:288`
- `_init_ep_weight_filter(...)`：`code/vllm/vllm/model_executor/model_loader/default_loader.py:318`
- `load_weights(...)`：`code/vllm/vllm/model_executor/model_loader/default_loader.py:382`

职责：

1. 根据 `load_format` 和模型路径准备权重文件；
2. 支持本地/HF/ModelScope 下载；
3. 过滤重复 safetensors / 非推理文件；
4. 选择 safetensors、pt、npcache、fastsafetensors、instanttensor 等 iterator；
5. 合并 primary / secondary weights；
6. 初始化 EP weight filter；
7. 调用 `model.load_weights(weights_iterator)`。

一句话：`DefaultModelLoader` 是标准 checkpoint 加载的总装路径，覆盖大多数模型。

## 6. `BitsAndBytesModelLoader`

类定义附近：`code/vllm/vllm/model_executor/model_loader/bitsandbytes_loader.py:56`。

关键入口：

- `_prepare_weights(...)`：`code/vllm/vllm/model_executor/model_loader/bitsandbytes_loader.py:119`
- `_get_quantized_weights_iterator(...)`：`code/vllm/vllm/model_executor/model_loader/bitsandbytes_loader.py:191`
- `load_weights(...)`：`code/vllm/vllm/model_executor/model_loader/bitsandbytes_loader.py:804`

职责：

- 加载 BnB 4bit / 8bit 权重；
- 区分量化张量和非量化张量；
- 初始化 loader state；
- 维护 quant state；
- 处理 MoE quant states 的 fuse / stack / bind；
- 做模型兼容性检查。

它和默认 loader 的区别是：BnB 加载不只是读 tensor，还要恢复量化元状态。

## 7. `ShardedStateLoader`

类定义：`code/vllm/vllm/model_executor/model_loader/sharded_state_loader.py:29`。

加载入口：`code/vllm/vllm/model_executor/model_loader/sharded_state_loader.py:110`。

特点：

- 面向已经按 rank 分片的 checkpoint；
- 默认命名模式类似 `model-rank-{rank}-part-{part}.safetensors`；
- 当前 rank 只读自己的 shard；
- `_filter_subtensors(...)` 用于过滤子张量/重复 view；
- 还提供 `save_model(...)`，说明它支持保存预分片 checkpoint。

适用场景：大模型分布式部署时，不希望每个 rank 都扫描/读取全量 checkpoint。

## 8. `TensorizerLoader`

类定义：`code/vllm/vllm/model_executor/model_loader/tensorizer_loader.py:43`。

关键入口：

- `load_weights(...)`：`code/vllm/vllm/model_executor/model_loader/tensorizer_loader.py:103`
- `load_model(...)`：`code/vllm/vllm/model_executor/model_loader/tensorizer_loader.py:115`

特点：

- 面向 tensorizer 序列化格式；
- 可以直接按 tensorizer 的序列化格式恢复模型；
- `load_model()` 有专门实现，不完全依赖 `BaseModelLoader` 模板；
- 包含 `_patch_tensorizer_config(...)` 等格式适配逻辑。

一句话：它更像 tensorizer 格式原生恢复器。

## 9. `RunaiModelStreamerLoader`

类定义：`code/vllm/vllm/model_executor/model_loader/runai_streamer_loader.py:21`。

关键入口：

- `_prepare_weights(...)`：`code/vllm/vllm/model_executor/model_loader/runai_streamer_loader.py:72`
- `_get_weights_iterator(...)`：`code/vllm/vllm/model_executor/model_loader/runai_streamer_loader.py:112`
- `load_weights(...)`：`code/vllm/vllm/model_executor/model_loader/runai_streamer_loader.py:125`

特点：

- 面向 RunAI 远程 streaming 权重读取；
- `extra_config` 可包含 `distributed`、`concurrency`、`memory_limit`；
- 重点优化远程对象存储/流式读取，而不是本地 checkpoint 扫描。

## 10. `DummyModelLoader`

类定义：`code/vllm/vllm/model_executor/model_loader/dummy_loader.py:22`。

入口：`code/vllm/vllm/model_executor/model_loader/dummy_loader.py:36`。

特点：

- 不加载真实 checkpoint；
- 构造 dummy weights；
- 常用于性能测试、图编译预热、某些 online quant 物化路径、快速初始化模型壳。

它适合验证 runtime 路径，不适合验证模型精度。

## 11. `ModelExpressModelLoader`

类定义：`code/vllm/vllm/model_executor/model_loader/modelexpress_loader.py:33`。

入口：`code/vllm/vllm/model_executor/model_loader/modelexpress_loader.py:59`。

特点：

- 自身不实现底层 checkpoint 解析；
- 委托给外部 `modelexpress.engines.vllm.loader`；
- 是外部生态的适配包装器。

## 12. loader 选择和配置的关系

loader 选择主要由 `LoadConfig.load_format` 决定，但 loader 内部行为还会读取其他配置：

| 配置 | 对 loader 的影响 |
|---|---|
| `ModelConfig.model` | 模型路径或 HF repo。 |
| `ModelConfig.revision` | 下载/读取哪个版本。 |
| `ModelConfig.quantization` | 量化 loader 或量化后处理。 |
| `ModelConfig.dtype` | 初始化模型和权重加载 dtype context。 |
| `ParallelConfig` | rank-aware 权重过滤、sharded checkpoint、EP filter。 |
| `LoadConfig.download_dir` | 下载缓存目录。 |
| `LoadConfig.ignore_patterns` | 文件发现时过滤。 |
| `LoadConfig.model_loader_extra_config` | RunAI / ModelExpress 等特殊 loader 参数。 |

## 13. 一句话总结

`LoadConfig` 决定“用什么方式读权重”，`get_model_loader()` 把 `load_format` 映射到具体 loader，`BaseModelLoader` 统一模型实例化与后处理生命周期，而各 loader 子类负责不同 checkpoint 格式、量化状态、远程流式读取或预分片加载的具体细节。
