# 12 lora_and_adapters 背诵文档

## 1. 专题定位

`lora_and_adapters` 讲的是 vLLM 中 LoRA adapter 如何动态加载、请求级选择、batch 内混合执行，并在 forward 中叠加到 base model 上。

LoRA 和 base model 不同：

```text
base model：Worker 初始化时加载，所有请求共享。
LoRA adapter：可以动态加载、卸载、pin，请求级选择，batch 内可混合多个 adapter。
```

一句话：

```text
vLLM 的 LoRA 是一套请求级可切换的 adapter 执行系统。
```

## 2. 最小心智模型

请求执行主线：

```text
用户请求携带 LoRARequest
  → InputProcessor._validate_lora()
  → EngineCoreRequest.lora_request
  → Request.lora_request
  → SchedulerOutput.NewRequestData.lora_request
  → GPUModelRunner._update_states()
  → CachedRequestState.lora_request
  → InputBatch.request_lora_mapping
  → GPUModelRunner._prepare_inputs()
  → InputBatch.make_lora_inputs()
  → LoRAModelRunnerMixin.set_active_loras()
  → WorkerLoRAManager.set_active_adapters()
  → LoRAModelManager.set_adapter_mapping()
  → punica_wrapper.update_metadata()
  → LoRA layer forward
  → base output + LoRA delta
```

控制面主线：

```text
Executor.add_lora() / remove_lora() / pin_lora() / list_loras()
  → collective_rpc 到所有 Worker
  → Worker 转发给 ModelRunner
  → LoRAModelRunnerMixin
  → WorkerLoRAManager
  → LoRAModelManager / adapter cache
```

## 3. LoRA 分层

vLLM LoRA 横跨多层：

```text
配置层：LoRAConfig 定义 max_loras、rank、dtype、target_modules 等能力边界。
请求层：LoRARequest 表示这个请求想用哪个 adapter。
Engine / Scheduler 层：携带 LoRARequest，不执行 LoRA forward。
Worker / ModelRunner 层：把 request 级 LoRA 映射成 batch / token 级 active mapping。
Manager 层：加载、缓存、激活、卸载 adapter 权重。
Layer 层：LoRA-wrapped Linear / Embedding / LM head / MoE 叠加 delta。
Kernel / metadata 层：punica wrapper 组织 batch mixed LoRA 计算。
控制面：add_lora / remove_lora / pin_lora / list_loras。
```

一句话：

```text
LoRARequest 说请求想用哪个 adapter；InputBatch 说本轮每个 token 用哪个 adapter；LoRA manager 保证权重可用；LoRA layer 在 forward 中按 mapping 计算 delta。
```

## 4. LoRAConfig

`LoRAConfig` 定义 LoRA 能力边界。

核心字段：

```text
max_lora_rank
max_loras
fully_sharded_loras
max_cpu_loras
lora_dtype
target_modules
default_mm_loras
enable_tower_connector_lora
specialize_active_lora
enable_mixed_moe_lora_format
```

容量相关：

```text
max_loras：单个 batch 最多同时 active 的 LoRA 数量。
max_cpu_loras：CPU 侧最多缓存多少 LoRA，必须 >= max_loras。
max_lora_rank：支持的最大 LoRA rank。
```

执行图相关：

```text
max_lora_rank / max_loras / fully_sharded_loras / lora_dtype / target_modules 等会影响计算图 hash。
```

这说明：

```text
LoRA 配置不是纯元数据，它会影响模型包装、kernel 路径和 CUDA graph。
```

## 5. LoRARequest

`LoRARequest` 是请求级 adapter 选择信息。

核心字段：

```text
lora_name
lora_int_id
lora_path
base_model_name
tensorizer_config_dict
load_inplace
is_3d_lora_weight
```

重要约定：

```text
lora_int_id 必须 > 0。
InputBatch 中 0 表示 no-LoRA。
```

因此 mapping 中：

```text
0：base model only。
>0：某个 LoRA adapter id。
```

还要注意：

```text
LoRARequest.__hash__ 通常按 lora_name 去重。
```

## 6. 请求如何进入 LoRA 链路

请求进入时：

```text
InputProcessor._validate_lora(lora_request)
```

它会检查：

```text
如果请求携带 LoRARequest，但 engine 没启用 LoRA，则报错。
```

之后 LoRARequest 被保存到：

```text
EngineCoreRequest.lora_request
  → Request.lora_request
  → SchedulerOutput.NewRequestData.lora_request
  → CachedRequestState.lora_request
```

Scheduler 本身不执行 LoRA forward。

它只携带 request 上的 LoRARequest，并在调度时考虑 active LoRA 数量限制。

## 7. Worker / ModelRunner LoRA 状态

`GPUModelRunner._update_states()` 把新请求转成 worker 侧 `CachedRequestState`，保存：

```text
lora_request
```

`InputBatch.add_request()` 记录：

```text
request_lora_mapping[req_index] = lora_id
lora_id_to_request_ids[lora_id].add(req_id)
lora_id_to_lora_request[lora_id] = LoRARequest
```

没有 LoRA 的请求：

```text
request_lora_mapping[req_index] = 0
```

一句话：

```text
InputBatch 把请求级 LoRA 状态变成 batch 级 LoRA mapping。
```

## 8. make_lora_inputs

每轮 token 数确定后，ModelRunner 调：

```text
set_active_loras(input_batch, num_scheduled_tokens, num_sampled_tokens)
```

`InputBatch.make_lora_inputs()` 把 request 级 mapping 展开成 token 级 mapping：

```text
request_lora_mapping：req_index → lora_id

token_lora_mapping：flat scheduled token index → lora_id
prompt_lora_mapping：sampled / logits 相关位置 → lora_id
active_lora_requests：本轮需要的 LoRARequest 集合
```

例子：

```text
request_lora_mapping = [0, A, A, B]
num_scheduled_tokens = [1, 3, 2, 1]

token_lora_mapping = [0, A, A, A, A, A, B]
```

## 9. LoRAModelRunnerMixin

`GPUModelRunner` 继承 LoRA mixin。

它提供两类能力：

### 模型加载阶段

```text
load_lora_model()
  → 创建 LRUCacheWorkerLoRAManager
  → manager 包装模型
  → 替换目标 layer 为 LoRA wrapper
```

在 `GPUModelRunner.load_model()` 中：

```text
if lora_config:
    self.model = self.load_lora_model(...)
```

### 执行前 active mapping

```text
set_active_loras()
  → InputBatch.make_lora_inputs()
  → LoRAMapping(index_mapping, prompt_mapping, type)
  → lora_manager.set_active_adapters(lora_requests, lora_mapping)
```

一句话：

```text
LoRAModelRunnerMixin 把 ModelRunner 的 batch 状态下发给 LoRA manager。
```

## 10. WorkerLoRAManager

`WorkerLoRAManager` 是 Worker 侧 adapter 管理器。

它负责：

```text
加载 LoRA checkpoint
创建 LoRAModelManager
add_adapter / remove_adapter / pin_adapter / list_adapters
set_active_adapters
根据当前请求集合应用 adapter
```

`set_active_adapters()` 做两件事：

```text
1. _apply_adapters(requests)：确保本轮需要的 adapter 权重已加载 / 激活。
2. set_adapter_mapping(mapping)：把本轮 token mapping 交给底层 model manager。
```

## 11. LRUCacheWorkerLoRAManager

它继承 WorkerLoRAManager。

作用：

```text
用 LRU cache 管理 adapter。
允许 CPU 侧缓存数量和 GPU active slot 数量不同。
```

典型限制：

```text
GPU active adapter 数不能超过 max_loras。
CPU cached adapter 数不能超过 max_cpu_loras。
```

## 12. LoRAModelManager

`LoRAModelManager` 更接近模型结构。

它负责：

```text
初始化 punica wrapper
遍历并替换模型中的 LoRA target modules
维护 LoRA-wrapped modules
把 adapter 权重复制到各 layer 的 LoRA slot
维护 lora_index_to_id
更新 punica wrapper metadata
```

`activate_adapter()`：

```text
把某个 LoRA id 放入 GPU slot，并把权重写入所有相关 LoRA layer。
```

`set_adapter_mapping()`：

```text
当本轮 token mapping 改变时，更新 punica metadata。
```

## 13. LoRA layer 注入

LoRA 注入发生在模型加载阶段，不是每次 forward 临时替换。

主链路：

```text
GPUModelRunner.load_model()
  → load_lora_model()
  → WorkerLoRAManager.create_lora_manager()
  → LoRAModelManager.__init__()
  → _init_punica_wrapper()
  → _create_lora_modules()
  → 遍历 model.named_modules()
  → from_layer(...) 选择 LoRA wrapper
  → 替换原 module
  → wrapper.create_lora_weights()
  → wrapper.set_mapping(punica_wrapper)
```

常见被包装层：

```text
ColumnParallelLinear
RowParallelLinear
ReplicatedLinear
MergedColumnParallelLinear
QKVParallelLinear
VocabParallelEmbedding
LogitsProcessor / LM head
FusedMoE / MoE runner
```

## 14. batch mixed LoRA

vLLM 支持同一个 batch 混合多个 LoRA。

例子：

```text
req0：no LoRA
req1：LoRA A
req2：LoRA A
req3：LoRA B
```

forward 时逻辑公式：

```text
output = base_output + x @ A @ B * scaling
```

工程上还要处理：

```text
token 到 LoRA id 的映射
LoRA id 到 GPU slot index 的映射
no-LoRA token
同 batch 多 adapter
prefill / decode / spec decode token 数差异
embedding / logits processor 特殊 mapping
packed linear / MoE / TP 分片
```

一句话：

```text
batch mixed LoRA 的核心是 token_lora_mapping 和 GPU slot metadata。
```

## 15. 控制面和请求执行面的区别

### 控制面

决定 adapter 是否存在：

```text
add_lora
remove_lora
pin_lora
list_loras
```

路径：

```text
Executor
  → collective_rpc 到所有 Worker
  → Worker
  → ModelRunner
  → WorkerLoRAManager
```

### 请求执行面

决定本轮哪些 token 用哪个 adapter：

```text
LoRARequest
  → InputBatch.request_lora_mapping
  → LoRAMapping
  → punica metadata
  → LoRA layer forward
```

两者都正确，LoRA 才能运行。

## 16. 多模态 LoRA

多模态中有两条 LoRA 路径。

### language model LoRA

默认路径。

```text
多模态 encoder output 先变成 embeddings。
language model 部分使用 LoRAMappingType.LANGUAGE。
```

### tower / connector LoRA

如果：

```text
enable_tower_connector_lora=True
```

并且模型支持，LoRA 也可能作用在：

```text
vision/audio tower
multimodal connector
```

这时 mapping 单位不是 language token，而是多模态 item / encoder batch。

对应 mapping 类型：

```text
LoRAMappingType.TOWER
LoRAMappingType.CONNECTOR
```

重要点：

```text
tower connector LoRA 会影响 encoder cache key。
同一张图片在不同 LoRA 下可能产生不同 embedding。
```

## 17. LoRA 与 Tensor Parallel

LoRA wrapper 必须适配 base layer 的并行方式。

例如：

```text
ColumnParallelLinear：输出维分片。
RowParallelLinear：输入维分片。
QKV / gate_up packed layer：多个 logical module 合并。
```

`fully_sharded_loras` 会改变 TP 下 LoRA 计算切分策略。

一句话：

```text
LoRA 权重也要按当前 rank 的 base layer 分片方式加载和执行。
```

## 18. LoRA 与 Pipeline Parallel

PP 下不是每个 rank 都有完整模型层。

因此：

```text
LoRA layer 注入只发生在当前 PP stage 实际存在的 module 上。
不存在的 layer 不应加载 LoRA 权重。
```

## 19. LoRA 与 Data Parallel

DP 下每个副本都要有一致 adapter 集合。

Executor 通过 collective RPC 广播：

```text
add_lora / remove_lora / pin_lora / list_loras
```

`list_loras()` 通常要求所有 worker 返回一致集合。

## 20. LoRA 与量化

LoRA delta 通常叠加在量化 base layer 输出上。

关键点：

```text
base weight 可以是 GPTQ / AWQ / compressed / FP8。
LoRA A/B 通常按 lora_dtype 加载。
LoRA wrapper 要知道 base layer 的实际设备、并行和权重布局。
```

不是所有 LoRA + 量化组合都必然支持。

## 21. LoRA 与 CUDA graph

LoRA active 数量会影响 CUDA graph dispatch。

ModelRunner 会计算：

```text
num_active_loras
has_lora
```

这些可能进入：

```text
BatchDescriptor
CUDA graph key
graph specialization
```

因此：

```text
不同 active LoRA 数量可能需要不同 graph，或导致 fallback。
```

## 22. warmup / dummy LoRA

LoRA 会影响 kernel 和 CUDA graph。

warmup / capture 阶段需要模拟 LoRA batch。

常见能力：

```text
maybe_setup_dummy_loras()
maybe_select_dummy_loras()
maybe_dummy_run_with_lora()
maybe_remove_all_loras()
```

用途：

```text
创建 dummy LoRARequest
构造 dummy mapping
覆盖 LoRA graph / kernel 路径
warmup 后清理
```

## 23. 关键数据结构关系

```text
LoRAConfig：配置能力边界和执行图参数。
LoRARequest：请求级 adapter 选择。
EngineCoreRequest.lora_request：Engine 侧携带。
Request.lora_request：Scheduler 侧携带。
NewRequestData.lora_request：SchedulerOutput 下发给 Worker。
CachedRequestState.lora_request：Worker 侧请求状态。
InputBatch.request_lora_mapping：req_index → lora_id。
InputBatch.lora_id_to_lora_request：lora_id → LoRARequest。
LoRAMapping：token / prompt mapping + type。
lora_index_to_id：GPU slot index → LoRA id。
punica wrapper metadata：LoRA layer forward 直接使用的 mixed LoRA metadata。
```

## 24. 常见约束和易错点

```text
lora_int_id 不能为 0，因为 0 表示 no-LoRA。
batch 内 active LoRA 数不能超过 max_loras。
max_cpu_loras 必须 >= max_loras。
LoRARequest 只是选择信息，不代表 adapter 已加载。
active adapter 和 active mapping 是两件事。
多模态 tower LoRA 会影响 encoder cache key。
同名 LoRARequest 可能按 lora_name 去重。
```

## 25. LoRARequest 不等于 adapter 已加载

要特别背：

```text
LoRARequest 只是请求说“我要用这个 adapter”。
真正加载发生在 WorkerLoRAManager.add_adapter() / _apply_adapters() / activate_adapter()。
```

如果请求携带 LoRARequest，但 worker 没有加载或无法加载 adapter，执行仍会失败。

## 26. active adapter 和 active mapping

两者不同：

```text
active adapter：权重在 manager / GPU slot 中可用。
active mapping：本轮 token 使用哪个 adapter。
```

有 adapter 不代表本轮 token 使用它。

有 mapping 也要求 adapter 已被 manager 激活。

## 27. 与其他专题的关系

```text
config_and_model_loading：LoRAConfig 如何进入 VllmConfig，load_model 时如何包装模型。
model_architectures：SupportsLoRA、packed_modules_mapping、embedding_modules 决定可注入位置。
executor_worker_model_runner：ModelRunner 如何维护 LoRA batch state。
scheduler：调度时如何考虑 max_loras 限制。
parallelism：TP / PP / DP 下 LoRA 权重和控制面如何分布。
quantization：LoRA 如何叠加在量化 base layer 上。
multimodal：tower / connector LoRA 和 encoder cache key。
compilation_and_cuda_graph：active LoRA 数量影响 graph key 和 fallback。
```

## 28. 背诵总结

背这一段：

```text
vLLM 的 LoRA 是请求级可切换 adapter 系统。base model 在 Worker 初始化时固定加载，LoRA adapter 可以通过控制面动态 add、remove、pin 和 list。请求用 LoRARequest 指定 adapter，Engine 和 Scheduler 只携带这个信息；Worker / ModelRunner 把 request 级 LoRARequest 写入 CachedRequestState 和 InputBatch，再根据本轮 num_scheduled_tokens 展开成 token_lora_mapping，交给 LoRA manager。LoRA manager 保证 adapter 权重已加载到 GPU slot，并更新 punica metadata；LoRA-wrapped Linear、Embedding、LM head 或 MoE layer 在 forward 中计算 base output，并按 token mapping 叠加 LoRA delta。LoRA 的关键边界是：LoRARequest 是选择信息，adapter manager 负责权重可用，InputBatch 负责本轮 token mapping，LoRA layer 负责实际 delta 计算。
```
