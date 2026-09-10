# 09 Worker 到 ModelRunner 加载链路

本篇梳理 vLLM V1 runtime 中 worker 如何触发模型加载。前几篇关注配置、registry、loader；本篇把这些组件放回 runtime，看 `Worker.load_model()`、`GPUModelRunner.load_model()`、reload、post-process 如何串起来。

## 1. 总体链路

```text
Executor 初始化 worker
  ↓
Worker.load_model(load_dummy_weights=False)
  ↓
GPUModelRunner.load_model(load_dummy_weights=False)
  ↓
get_model_loader(self.load_config)
  ↓
model_loader.load_model(vllm_config, model_config)
  ↓
initialize_model(...)
  ↓
loader.load_weights(...)
  ↓
process_weights_after_loading(...)
  ↓
GPUModelRunner.model = model
```

关键入口：

- `Worker.load_model()`：`code/vllm/vllm/v1/worker/gpu_worker.py:349`
- `Worker.get_model()`：`code/vllm/vllm/v1/worker/gpu_worker.py:760`
- `GPUModelRunner.get_model()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3198`
- `GPUModelRunner.load_model()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5143`
- loader 调用位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5163`
- `GPUModelRunner.reload_weights()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5376`

## 2. `Worker.load_model()` 的职责

入口：`code/vllm/vllm/v1/worker/gpu_worker.py:349`。

`Worker` 是 runtime 侧对象，职责不是解析模型文件，而是管理当前 worker 的执行上下文。

它通常负责：

- 确认当前 worker 所在设备；
- 管理显存池 / memory snapshot；
- 进入 allocator 或 device context；
- 调用 `model_runner.load_model(...)`；
- 在加载前后更新内存统计；
- 将模型加载和后续 KV cache 初始化、profile、执行流程衔接。

可以理解为：

```text
Worker.load_model()
  = runtime/device/memory wrapper
  + delegate to ModelRunner.load_model()
```

## 3. `GPUModelRunner.load_model()` 的职责

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5143`。

`GPUModelRunner` 是模型执行链路的核心对象。加载阶段它负责把 `VllmConfig` 中的配置传给 model loader，并保存返回的 `nn.Module`。

简化流程：

```text
GPUModelRunner.load_model(load_dummy_weights)
  ↓
如果 load_dummy_weights=True，临时使用 dummy loader/配置
  ↓
model_loader = get_model_loader(self.load_config)
  ↓
self.model = model_loader.load_model(
      vllm_config=self.vllm_config,
      model_config=self.model_config,
  )
  ↓
后续 runner 可通过 self.model 执行 forward
```

loader 调用位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5163`。

## 4. `load_dummy_weights` 的意义

`load_dummy_weights=True` 时，runner 不加载真实 checkpoint，而是构造 dummy 权重。

常见用途：

- profile 内存；
- cudagraph/compile 预热；
- 测试 runner 管线；
- 某些量化或延迟加载路径；
- 在没有真实权重时验证模型结构。

它会让系统走类似加载链路，但不代表真实模型精度可用。

## 5. `BaseModelLoader.load_model()` 回到加载模板

`GPUModelRunner` 并不直接读文件。它把工作交给 loader。

通用模板：`code/vllm/vllm/model_executor/model_loader/base_loader.py:42`。

```text
BaseModelLoader.load_model(...)
  ↓
with set_default_torch_dtype(model_config.dtype)
with target_device context
  ↓
initialize_model(vllm_config, prefix="")
  ↓
self.load_weights(model, model_config)
  ↓
finalize_layerwise_processing(...)
  ↓
process_weights_after_loading(...)
  ↓
return model.eval()
```

这说明 runtime 的加载入口和具体 checkpoint 解析之间有一层清晰边界：runner 不知道 safetensors/bin/RunAI/BnB 细节。

## 6. post-process 阶段

权重加载后处理入口：`code/vllm/vllm/model_executor/model_loader/utils.py:100`。

设备上下文入口：`code/vllm/vllm/model_executor/model_loader/utils.py:134`。

post-process 可能包括：

- quantization module finalize；
- attention layer 权重后处理；
- KV cache scale 默认值修正；
- FP8 scale reshape；
- MoE 权重重排；
- LoRA 相关状态；
- backend-specific preprocess；
- 多模态 encoder 后处理。

因此“权重读完”不等于“模型可运行”，必须经过后处理。

## 7. reload 权重链路

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5376`。

简化逻辑：

```text
GPUModelRunner.reload_weights(weights_iterator=None, is_checkpoint_format=True/False)
  ↓
如果 weights_iterator is None
  → 通过当前 loader 重新 get_all_weights(...)
  ↓
如果 is_checkpoint_format=True
  → initialize_layerwise_reload(model)
  → model.load_weights(weights_iterator)
  → finalize_layerwise_reload(model, model_config)
否则
  → 按 name 直接找 parameter 并 copy_
```

reload 相关函数：

- `record_metadata_for_reloading()`：`code/vllm/vllm/model_executor/model_loader/reload/layerwise.py:70`
- `initialize_layerwise_reload()`：`code/vllm/vllm/model_executor/model_loader/reload/layerwise.py:84`
- `finalize_layerwise_processing()`：`code/vllm/vllm/model_executor/model_loader/reload/layerwise.py:217`
- `finalize_layerwise_reload()`：`code/vllm/vllm/model_executor/model_loader/reload/layerwise.py:276`

## 8. reload 的两种语义

### 8.1 checkpoint-aware reload

```text
is_checkpoint_format=True
  ↓
model.load_weights(weights_iterator)
```

这种路径保留模型类自定义加载语义，支持：

- QKV fused 映射；
- TP/PP/EP 过滤；
- quantization scale；
- tied embedding；
- checkpoint 命名兼容。

适合从 checkpoint 重新加载权重。

### 8.2 raw tensor copy reload

```text
is_checkpoint_format=False
  ↓
按 name 找 parameter
  ↓
param.copy_(tensor)
```

这种路径更直接，但能力较弱。它要求传入名称基本已经对应当前模型参数名，不再经过模型类复杂映射逻辑。

## 9. `get_model()` 查询

worker/model runner 提供 `get_model()`：

- `code/vllm/vllm/v1/worker/gpu_worker.py:760`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3198`

它们通常用于：

- LoRA 管理；
- profiling；
- inspect/debug；
- speculative decode；
- 外部控制接口；
- 权重 reload；
- 获取模型模块做后续操作。

注意：只有 `load_model()` 完成后，`get_model()` 返回的模型才真正可用于 forward。

## 10. 和 executor / distributed 的关系

在分布式场景中，每个 worker/rank 都会加载模型，但加载结果不一定相同：

- TP rank 加载不同 tensor shard；
- PP rank 加载不同层段；
- EP rank 加载不同 experts；
- DP replica 加载一份完整或相同分片集合；
- sharded checkpoint 下每个 rank 直接读取自己的 shard；
- Ray / multiprocessing worker 需要可序列化的 `VllmConfig`。

所以 `Worker.load_model()` 是每个 runtime worker 的本地行为，不是单进程全局加载一次。

## 11. 加载和 KV cache 初始化的边界

模型加载主要负责创建 `nn.Module` 并加载权重。KV cache 的实际分配、block 管理、profile 可用 block 数等是后续 worker/cache manager 的职责。

但二者高度相关：

- 模型层数、KV heads、head size 来自 `ModelConfig.model_arch_config`；
- cache dtype/block size 来自 `CacheConfig`；
- runner 需要模型结构来 profile 内存；
- cudagraph/compile 需要模型已经可 forward。

因此模型加载完成通常是 KV cache/profile/compile 的前置条件。

## 12. 加载和 forward 的边界

`GPUModelRunner.load_model()` 完成后，runner 才能在请求执行时调用模型 forward。

后续 forward 会依赖加载阶段建立的结构：

- attention layer 已注册；
- quantized weight 已 finalize；
- MoE 权重已重排；
- LoRA hooks/状态可用；
- 多模态 encoder/projector 已初始化；
- sampler/logits processor 可用；
- PP intermediate tensor 接口可用。

因此加载链路的问题经常会在第一次 forward 时才暴露。

## 13. 常见定位问题

### 13.1 worker 报 OOM

检查：

```text
Worker.load_model 前后显存统计
model_config.dtype
load_dummy_weights 是否启用
quantization 是否生效
parallel_config 是否按预期分片
GPU memory utilization / offload 配置
```

### 13.2 单机能加载，多卡失败

重点检查：

```text
ParallelConfig TP/PP/EP/DP
分布式初始化是否完成
当前 rank 读取的 checkpoint shard
EP weight filter
模型类 load_weights 是否支持该并行组合
```

### 13.3 reload 后结果异常

确认：

```text
is_checkpoint_format 是否正确
传入 weights_iterator 的 name 是否是 checkpoint key 还是 parameter name
是否经过 initialize/finalize_layerwise_reload
量化状态是否同步更新
```

## 14. 一句话总结

在 runtime 中，`Worker.load_model()` 负责设备和内存上下文，`GPUModelRunner.load_model()` 负责连接配置与 loader，`BaseModelLoader` 负责实例化和后处理生命周期，具体 loader 和模型类共同完成权重读取与参数映射；reload 则在同一套结构上提供 checkpoint-aware 和 raw copy 两种更新语义。
