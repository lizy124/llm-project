# vLLM LoRA / Adapters 逻辑梳理

源码位置：

- `code/vllm/vllm/config/lora.py`
- `code/vllm/vllm/lora/request.py`
- `code/vllm/vllm/lora/worker_manager.py`
- `code/vllm/vllm/lora/model_manager.py`
- `code/vllm/vllm/lora/lora_model.py`
- `code/vllm/vllm/lora/lora_weights.py`
- `code/vllm/vllm/lora/utils.py`
- `code/vllm/vllm/lora/layers/`
- `code/vllm/vllm/lora/punica_wrapper/`
- `code/vllm/vllm/adapter_commons/`
- `code/vllm/vllm/v1/engine/`
- `code/vllm/vllm/v1/core/sched/`
- `code/vllm/vllm/v1/executor/`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py`

说明：当前代码库里，旧文档常见的 `vllm/lora/models.py`、`vllm/lora/layers.py` 已经拆分为 `lora_model.py`、`model_manager.py`、`lora_weights.py` 和 `lora/layers/*`。本文按当前 `code/vllm` 代码结构梳理。

本文按“先定边界，再走请求到 worker 主链路，再拆 adapter 管理、layer 注入、batch mixed 执行、控制面和限制”的方式，梳理 vLLM 中 LoRA / adapter 机制。

LoRA 与普通模型权重不同：

```text
base model：
  在 worker 初始化时加载，所有请求共享。

LoRA adapter：
  可以动态加载、卸载、pin，
  请求级选择，
  batch 内可混合多个 adapter，
  forward 时作为低秩 delta 叠加到 base layer 上。
```

---

## 0. 梳理规划

本目录要回答的问题分成 11 组：

```text
1. LoRA / adapter 在 vLLM 中处于哪一层？
2. LoRARequest 如何从 API / Engine 进入 Scheduler 和 Worker？
3. LoRA manager 如何加载、缓存、pin、卸载 adapter？
4. Worker / ModelRunner 如何维护当前 batch 的 active LoRA 状态？
5. LoRA layer 如何注入 Linear / Embedding / LM head / MoE？
6. 同一个 batch 中混合多个 LoRA 请求如何执行？
7. LoRA 权重如何加载、命名映射和切分？
8. LoRA 如何与量化 base model 共存？
9. LoRA 如何与 TP / PP / DP / 多模态并行交互？
10. add_lora / remove_lora / pin_lora / list_loras 生命周期如何走？
11. 常见限制、错误和调试入口有哪些？
```

阅读顺序建议：

```text
lora_and_adapters_overview.md
  → 01_lora_role.md
  → 02_lora_request_and_engine_flow.md
  → 03_lora_manager_and_cache.md
  → 04_worker_model_runner_lora_state.md
  → 05_lora_layer_injection.md
  → 06_batch_mixed_lora_execution.md
  → 07_lora_loading_and_weight_mapping.md
  → 08_lora_and_quantization.md
  → 09_lora_and_parallelism.md
  → 10_lora_lifecycle_and_control.md
  → 11_lora_limitations_and_debugging.md
```

如果只想先抓主线，可以先读：

```text
lora_and_adapters_overview.md
  → 02_lora_request_and_engine_flow.md
  → 04_worker_model_runner_lora_state.md
  → 05_lora_layer_injection.md
  → 06_batch_mixed_lora_execution.md
```

---

## 1. 一句话回答

vLLM 中的 LoRA 是一套“请求级可切换的 adapter 执行系统”。

```text
base model 固定加载，
LoRA adapter 动态加载和缓存，
请求通过 LoRARequest 指定 adapter，
ModelRunner 在 batch 执行前激活对应 LoRA，
LoRA layer 在 forward 中把低秩 delta 叠加到 base layer 输出。
```

最小主线是：

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

控制面主线是：

```text
Executor.add_lora() / remove_lora() / pin_lora() / list_loras()
  → collective_rpc 到所有 Worker
  → Worker 转发给 ModelRunner
  → LoRAModelRunnerMixin
  → WorkerLoRAManager
  → LoRAModelManager / adapter cache
```

---

## 2. LoRA 机制的几个层次

vLLM 的 LoRA 不是单个类完成，而是横跨多层。

```text
配置层：
  LoRAConfig 决定 max_loras、max_lora_rank、target_modules、dtype、tower connector 等能力边界。

请求层：
  LoRARequest 表达这个请求想使用哪个 adapter。

Engine / Scheduler 层：
  只携带 LoRARequest，不执行 LoRA forward。

Worker / ModelRunner 状态层：
  InputBatch 把 request 级 LoRARequest 转成 batch / token 级 active mapping。

manager 层：
  WorkerLoRAManager / LoRAModelManager 负责加载、缓存、激活、卸载 adapter 权重。

layer 层：
  LoRA-wrapped Linear / Embedding / LM head / MoE 在 forward 中叠加 delta。

kernel / metadata 层：
  punica wrapper 根据 token / prompt mapping 组织 batch mixed LoRA 计算。

控制面：
  Executor / Worker 暴露 add_lora、remove_lora、pin_lora、list_loras 等能力。
```

一句话区分：

```text
LoRARequest 负责“请求想用哪个 adapter”；
InputBatch 负责“本轮每个 token 用哪个 adapter”；
LoRA manager 负责“adapter 权重是否可用”；
LoRA layer / punica wrapper 负责“forward 中如何按 mapping 计算 delta”。
```

---

## 3. 关键配置：LoRAConfig

`LoRAConfig` 定义 LoRA 能力边界。

源码位置：`code/vllm/vllm/config/lora.py:31`

核心字段包括：

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

### 3.1 容量相关

```text
max_loras：
  单个 batch 中最多同时 active 的 LoRA 数量。

max_cpu_loras：
  CPU 侧最多缓存多少 LoRA，必须 >= max_loras。

max_lora_rank：
  支持的最大 LoRA rank。
```

配置校验在：

源码位置：`code/vllm/vllm/config/lora.py:108`

```text
if self.max_cpu_loras is None:
    self.max_cpu_loras = self.max_loras
elif self.max_cpu_loras < self.max_loras:
    raise ValueError(...)
```

### 3.2 执行图相关

`compute_hash()` 会把影响计算图结构的字段加入 hash。

源码位置：`code/vllm/vllm/config/lora.py:81`

包括：

```text
max_lora_rank
max_loras
fully_sharded_loras
lora_dtype
enable_tower_connector_lora
enable_mixed_moe_lora_format
target_modules
```

这说明：

```text
LoRA 配置不是纯运行时元数据，它会影响模型包装、kernel 路径和编译 / graph cache。
```

### 3.3 多模态相关

`enable_tower_connector_lora` 打开后，LoRA 不只作用在 language model，还可能作用于 multimodal tower / connector。

源码位置：`code/vllm/vllm/config/lora.py:62`

它是实验特性，当前只支持部分多模态模型。

---

## 4. LoRARequest 是什么

`LoRARequest` 是请求级 adapter 选择信息。

源码位置：`code/vllm/vllm/lora/request.py:8`

核心字段：

```text
lora_name: str
lora_int_id: int
lora_path: str
base_model_name: str | None
tensorizer_config_dict: dict | None
load_inplace: bool
is_3d_lora_weight: bool
```

几个关键点：

```text
- lora_int_id 必须 > 0；
- lora_path 不能为空；
- adapter_id 属性返回 lora_int_id；
- name 属性返回 lora_name；
- path 属性返回 lora_path；
- __eq__ / __hash__ 按 lora_name 判断。
```

源码位置：

- `code/vllm/vllm/lora/request.py:39`
- `code/vllm/vllm/lora/request.py:46`
- `code/vllm/vllm/lora/request.py:58`

重要约定：

```text
LoRARequest.lora_int_id >= 1；
InputBatch 中使用 0 表示 no-LoRA。
```

这就是为什么 mapping 里常见：

```text
0：base model only
>0：某个 LoRA adapter id
```

---

## 5. 请求进入执行链路

用户请求中的 `LoRARequest` 首先进入 engine 输入处理。

### 5.1 InputProcessor 校验

入口在：

源码位置：`code/vllm/vllm/v1/engine/input_processor.py:146`

```text
_validate_lora(lora_request)
```

它会检查：

```text
如果请求携带 LoRARequest，但 engine 没有启用 LoRA，则报错。
```

源码逻辑：

```text
if lora_request is None:
    return

if not self.lora_config:
    raise ValueError(...)
```

### 5.2 EngineCoreRequest 携带 LoRARequest

`EngineCoreRequest` 定义中包含：

源码位置：`code/vllm/vllm/v1/engine/__init__.py:86`

```text
lora_request: LoRARequest | None
```

`InputProcessor.process_inputs()` 最终构造 `EngineCoreRequest` 时传入：

源码位置：`code/vllm/vllm/v1/engine/input_processor.py:370`

```text
EngineCoreRequest(..., lora_request=lora_request, ...)
```

### 5.3 Scheduler 侧 Request 保存 LoRARequest

EngineCore 把 `EngineCoreRequest` 转成 scheduler 侧 `Request`。

这时 LoRA 仍然只是请求状态的一部分：

```text
EngineCoreRequest.lora_request
  → Request.lora_request
```

Scheduler 本身不执行 LoRA forward，也不加载 adapter。

### 5.4 SchedulerOutput 传给 Worker

新请求第一次被调度时，`SchedulerOutput.scheduled_new_reqs` 中的 `NewRequestData` 会携带 `lora_request`。

源码位置：`code/vllm/vllm/v1/core/sched/output.py:30`

```text
NewRequestData.lora_request
```

主线是：

```text
LoRARequest
  → EngineCoreRequest
  → Request
  → NewRequestData
  → GPUModelRunner._update_states()
```

---

## 6. Worker / ModelRunner active LoRA 状态

`GPUModelRunner._update_states()` 接收 `SchedulerOutput` 后，会把新请求转成 worker 侧 `CachedRequestState`。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1194`

其中包含：

```text
lora_request=new_req_data.lora_request
```

然后 `InputBatch.add_request()` 会把 request 级 LoRA 状态登记到 batch。

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:468`

核心状态是：

```text
request_lora_mapping[req_index] = lora_id
lora_id_to_request_ids[lora_id].add(req_id)
lora_id_to_lora_request[lora_id] = LoRARequest
```

如果请求没有 LoRA：

```text
request_lora_mapping[req_index] = 0
```

每轮 `_prepare_inputs()` 后段，当本轮每个 request 的 token 数确定后，会调用：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2193`

```text
self.set_active_loras(
    self.input_batch,
    num_scheduled_tokens,
    num_sampled_tokens,
)
```

`InputBatch.make_lora_inputs()` 把 request 级 mapping 展开成 token 级 mapping。

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:976`

```text
prompt_lora_mapping = tuple(req_lora_mapping.repeat(num_sampled_tokens))
token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))
active_lora_requests = set(self.lora_id_to_lora_request.values())
```

这一步把：

```text
req_index -> lora_id
```

翻译成：

```text
flat scheduled token index -> lora_id
flat sampled token index -> lora_id
```

---

## 7. LoRAModelRunnerMixin 的作用

`GPUModelRunner` 继承 `LoRAModelRunnerMixin`。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:418`

它提供两类能力。

### 7.1 模型加载阶段能力

`load_lora_model()` 会创建 `LRUCacheWorkerLoRAManager`，并让 manager 包装模型。

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:31`

```text
self.lora_manager = LRUCacheWorkerLoRAManager(...)
return self.lora_manager.create_lora_manager(model, vllm_config)
```

`GPUModelRunner.load_model()` 中会调用它：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5167`

```text
if self.lora_config:
    self.model = self.load_lora_model(...)
```

### 7.2 执行前 active mapping 能力

`set_active_loras()` 会调用 `InputBatch.make_lora_inputs()`，然后包装成 `LoRAMapping`。

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:73`

```text
LoRAMapping(
  index_mapping=token_lora_mapping,
  prompt_mapping=prompt_lora_mapping,
  is_prefill=True,
  type=LoRAMappingType.LANGUAGE,
)
```

最后调用：

```text
self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
```

这就是 Worker / ModelRunner 把 batch 状态下发给 LoRA manager 的主入口。

---

## 8. WorkerLoRAManager 和 LoRAModelManager

LoRA manager 分两层。

### 8.1 WorkerLoRAManager

源码位置：`code/vllm/vllm/lora/worker_manager.py:25`

它是 Worker 侧 adapter 管理器，负责：

```text
- 加载 LoRA checkpoint；
- 创建底层 LoRAModelManager；
- add_adapter / remove_adapter / pin_adapter / list_adapters；
- set_active_adapters；
- 根据当前请求集合应用 adapter。
```

`set_active_adapters()` 的核心是：

源码位置：`code/vllm/vllm/lora/worker_manager.py:183`

```text
self._apply_adapters(requests)
if mapping is not None:
    self._adapter_manager.set_adapter_mapping(mapping)
```

也就是两件事：

```text
1. 确保本轮需要的 adapter 权重已经加载 / 激活；
2. 把本轮 token mapping 交给底层 model manager。
```

### 8.2 LRUCacheWorkerLoRAManager

`LRUCacheWorkerLoRAManager` 继承 `WorkerLoRAManager`。

源码位置：`code/vllm/vllm/lora/worker_manager.py:231`

它使用 LRU cache 管理 adapter，允许 CPU 侧缓存数量和 GPU active slot 数量不同。

### 8.3 LoRAModelManager

源码位置：`code/vllm/vllm/lora/model_manager.py:64`

它负责更接近模型结构的事情：

```text
- 初始化 punica wrapper；
- 遍历并替换模型中的 LoRA target modules；
- 维护 LoRA-wrapped modules；
- 把 adapter 权重复制到各 layer 的 LoRA slot；
- 维护 lora_index_to_id；
- 更新 punica wrapper metadata。
```

`activate_adapter()` 会把某个 LoRA id 放入一个 GPU slot，并把权重写入所有相关 LoRA layer。

源码位置：`code/vllm/vllm/lora/model_manager.py:285`

`set_adapter_mapping()` 会在 mapping 改变时更新 metadata。

源码位置：`code/vllm/vllm/lora/model_manager.py:1139`

真正下发给 punica wrapper：

源码位置：`code/vllm/vllm/lora/model_manager.py:344`

```text
punica_wrapper.update_metadata(
    mapping,
    self.lora_index_to_id,
    self.lora_slots + 1,
    self.vocab_size,
)
```

---

## 9. LoRA layer 注入

LoRA 注入不是每次 forward 临时做，而是在模型加载阶段完成。

主链路是：

```text
GPUModelRunner.load_model()
  → LoRAModelRunnerMixin.load_lora_model()
  → WorkerLoRAManager.create_lora_manager()
  → create_lora_manager(...)
  → LoRAModelManager.__init__()
  → _init_punica_wrapper()
  → _create_lora_modules()
  → 遍历 model.named_modules()
  → from_layer(...) 选择 LoRA wrapper
  → 替换原 module
  → wrapper.create_lora_weights(...)
  → wrapper.set_mapping(punica_wrapper)
```

关键源码：

- `code/vllm/vllm/lora/model_manager.py:139`
- `code/vllm/vllm/lora/model_manager.py:375`
- `code/vllm/vllm/lora/model_manager.py:499`
- `code/vllm/vllm/lora/utils.py`
- `code/vllm/vllm/lora/layers/`

LoRA wrapper 主要覆盖这些类型：

```text
- ColumnParallelLinear
- RowParallelLinear
- ReplicatedLinear
- MergedColumnParallelLinear
- QKVParallelLinear
- VocabParallelEmbedding
- LogitsProcessor / LM head
- FusedMoE / MoE runner
```

不同 wrapper 的细节会在 `05_lora_layer_injection.md` 展开。

---

## 10. batch mixed LoRA 执行

vLLM 支持同一个 forward 中混合多个 LoRA adapter。

例如：

```text
batch:
  req0: no LoRA
  req1: LoRA A
  req2: LoRA A
  req3: LoRA B
```

`InputBatch` 会得到：

```text
request_lora_mapping = [0, A, A, B]
```

如果本轮 scheduled token 数是：

```text
num_scheduled_tokens = [1, 3, 2, 1]
```

则：

```text
token_lora_mapping = [0, A, A, A, A, A, B]
```

这个 mapping 会进入 `LoRAMapping.index_mapping`。

forward 时 LoRA layer 先计算 base output，再按 token mapping 计算对应 adapter 的 delta。

核心公式仍然是：

```text
output = base_output + x @ A @ B * scaling
```

但工程上需要处理：

```text
- token 到 LoRA id 的映射；
- LoRA id 到 GPU slot index 的映射；
- no-LoRA token；
- 同一 batch 多 adapter；
- prefill / decode / spec decode 的 token 数差异；
- embedding / logits processor 的特殊 mapping；
- packed linear / MoE / TP 分片。
```

这部分是 `06_batch_mixed_lora_execution.md` 的重点。

---

## 11. Executor / Worker 控制面

LoRA 不只随请求自动激活，也可以通过控制接口动态加载、卸载、pin 和查询。

Executor 抽象接口在：

源码位置：`code/vllm/vllm/v1/executor/abstract.py:292`

```text
add_lora(lora_request)
remove_lora(lora_id)
pin_lora(lora_id)
list_loras()
```

它们通过 collective RPC 分发给所有 worker。

例如：

```text
return all(self.collective_rpc("add_lora", args=(lora_request,)))
```

Worker 抽象接口在：

源码位置：`code/vllm/vllm/v1/worker/worker_base.py:165`

GPU Worker 实现只是转发给 ModelRunner。

源码位置：`code/vllm/vllm/v1/worker/gpu_worker.py:958`

```text
Worker.add_lora()
  → model_runner.add_lora()
  → LoRAModelRunnerMixin.add_lora()
  → WorkerLoRAManager.add_adapter()
```

控制面和请求执行面的区别是：

```text
控制面：
  决定 adapter 是否被加载 / 注册 / pin / 移除。

请求执行面：
  决定本轮 batch 的哪些 token 使用哪些 adapter。
```

两者都要正确，LoRA forward 才会正确。

---

## 12. 多模态 LoRA

多模态模型里 LoRA 有两条路径。

### 12.1 language model LoRA

这是默认路径。

多模态输入先经过 encoder / connector 形成 embedding，然后 language model 部分使用 `LoRAMappingType.LANGUAGE`。

这和文本模型的 LoRA active mapping 基本一致。

### 12.2 tower / connector LoRA

如果 `enable_tower_connector_lora=True`，并且模型支持 tower / connector LoRA，`GPUModelRunner._execute_mm_encoder()` 会为多模态 encoder batch 单独构造 LoRA mapping。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2941`

原因是：

```text
multimodal encoder batch 的单位是 image / video / audio item，
不是 language model 的 flat scheduled token batch，
所以不能复用 InputBatch.make_lora_inputs() 生成的 language token mapping。
```

它会使用：

```text
LoRAMappingType.TOWER
LoRAMappingType.CONNECTOR
```

同时，InputProcessor 生成多模态 hash 时会考虑 LoRARequest。

源码位置：`code/vllm/vllm/v1/engine/input_processor.py:165`

```text
如果 enable_tower_connector_lora=True，
mm identifier 会变成 f"{lora_name}:{mm_hash}"。
```

这样可以避免同一张图片在不同 LoRA 下错误复用 encoder cache。

---

## 13. 与并行、量化、CUDA graph 的关系

### 13.1 Tensor Parallel

LoRA wrapper 需要根据 base layer 的并行方式切分 LoRA A/B 权重。

例如：

```text
ColumnParallelLinear：输出维分片；
RowParallelLinear：输入维分片；
QKV / gate_up packed layer：多个 logical module 合并成一个 runtime wrapper。
```

`fully_sharded_loras` 会改变 TP 下 LoRA 计算的切分策略。

### 13.2 Pipeline Parallel

PP 下不是所有 rank 都持有完整模型层。

LoRA layer 注入和 adapter 权重加载必须跟随当前 rank 实际存在的 module；缺失 stage 不应加载不存在的 LoRA layer。

### 13.3 Data Parallel

DP 下每个 worker 副本都要有一致的 adapter 集合。

Executor 的 `add_lora/remove_lora/pin_lora/list_loras` 通过 collective RPC 广播到所有 worker，并在 `list_loras()` 中断言所有 worker 返回一致集合。

源码位置：`code/vllm/vllm/v1/executor/abstract.py:304`

### 13.4 量化

LoRA delta 通常叠加在量化 base layer 的输出路径上。

关键点是：

```text
base weight 可以是 GPTQ / AWQ / compressed tensor；
LoRA A/B 通常按 lora_dtype 加载；
wrapper 需要知道 base layer 的实际设备和权重布局。
```

### 13.5 CUDA graph

LoRA active 数量会影响 CUDA graph dispatch。

`GPUModelRunner._determine_batch_execution_and_padding()` 会计算：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3845`

```text
num_active_loras = len(self.input_batch.lora_id_to_lora_request)
has_lora = num_active_loras > 0
```

并把它传给 cudagraph dispatcher。

`specialize_active_lora` 也和不同 active LoRA 数量下的 graph / kernel specialization 相关。

---

## 14. warmup / dummy LoRA

LoRA 会影响 kernel 和 CUDA graph，所以 warmup / capture 阶段也要能模拟 LoRA batch。

`LoRAModelRunnerMixin` 提供：

```text
maybe_setup_dummy_loras()
maybe_select_dummy_loras()
maybe_dummy_run_with_lora()
maybe_remove_all_loras()
```

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:93`

这些函数会创建 dummy `LoRARequest`，构造 dummy mapping，并在 warmup 后清理。

GPU Worker warmup 中也会调用：

源码位置：`code/vllm/vllm/v1/worker/gpu_worker.py:621`

```text
self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)
```

---

## 15. 关键数据结构关系

### 15.1 `LoRAConfig`

配置 LoRA 能力边界和执行图相关参数。

### 15.2 `LoRARequest`

请求级 adapter 选择。

```text
lora_name + lora_int_id + lora_path
```

### 15.3 `EngineCoreRequest.lora_request` / `Request.lora_request`

Engine / Scheduler 侧携带 LoRA 信息。

### 15.4 `NewRequestData.lora_request`

SchedulerOutput 把新请求的 LoRARequest 传给 Worker。

### 15.5 `CachedRequestState.lora_request`

Worker 侧请求缓存状态中的 LoRARequest。

### 15.6 `InputBatch.request_lora_mapping`

```text
req_index -> lora_id
```

### 15.7 `InputBatch.lora_id_to_lora_request`

```text
lora_id -> LoRARequest
```

### 15.8 `LoRAMapping`

```text
index_mapping：本轮 forward token 到 LoRA id 的映射；
prompt_mapping：本轮 sampled/logits 相关位置到 LoRA id 的映射；
type：LANGUAGE / TOWER / CONNECTOR。
```

### 15.9 `lora_index_to_id`

`LoRAModelManager` 中的 GPU slot index 到 LoRA id 的映射。

### 15.10 punica wrapper metadata

LoRA layer forward 直接使用的 batch mixed LoRA metadata。

---

## 16. 常见约束和易错点

### 16.1 `lora_int_id` 不能为 0

0 被 `InputBatch` 用作 no-LoRA sentinel。

### 16.2 batch 内 active LoRA 数不能超过 `max_loras`

`WorkerLoRAManager._apply_adapters()` 会检查 requested adapter 数量是否超过 GPU adapter slots。

### 16.3 `max_cpu_loras` 必须 >= `max_loras`

否则配置校验会报错。

### 16.4 LoRARequest 只是选择信息，不代表 adapter 已经加载

真正加载发生在：

```text
WorkerLoRAManager.add_adapter()
  → _load_adapter()
  → LoRAModel.from_local_checkpoint(...)
  → LoRAModelManager.add_adapter()
  → activate_adapter()
```

### 16.5 active adapter 和 active mapping 是两件事

```text
active adapter：权重在 manager / GPU slot 中可用；
active mapping：本轮 token 使用哪个 adapter。
```

### 16.6 多模态 tower LoRA 会影响 encoder cache key

如果启用 tower / connector LoRA，同一多模态输入在不同 LoRA 下可能得到不同 embedding，因此 mm identifier 需要包含 LoRA 名称。

### 16.7 同名 LoRARequest 会被 set 去重

`LoRARequest.__hash__()` 按 `lora_name`，所以 `set[LoRARequest]` 的去重不是按路径或 id。

---

## 17. 推荐阅读路线

### 17.1 快速建立全局印象

```text
lora_and_adapters_overview.md
  → 01_lora_role.md
  → 02_lora_request_and_engine_flow.md
```

### 17.2 按请求到 forward 主链路阅读

```text
02_lora_request_and_engine_flow.md
  → 04_worker_model_runner_lora_state.md
  → 05_lora_layer_injection.md
  → 06_batch_mixed_lora_execution.md
```

### 17.3 按 adapter 生命周期阅读

```text
03_lora_manager_and_cache.md
  → 07_lora_loading_and_weight_mapping.md
  → 10_lora_lifecycle_and_control.md
  → 11_lora_limitations_and_debugging.md
```

### 17.4 按工程交互阅读

```text
08_lora_and_quantization.md
  → 09_lora_and_parallelism.md
  → 06_batch_mixed_lora_execution.md
```

---

## 18. 文档定位

```text
lora_and_adapters_overview.md：
  总览主文档，建立 LoRA 在 vLLM 中从请求到 forward 的全局图。

01_lora_role.md：
  定义 LoRA / adapter 的职责、边界和与 base model 的关系。

02_lora_request_and_engine_flow.md：
  梳理 LoRARequest 如何从用户请求进入 EngineCoreRequest、Request 和 SchedulerOutput。

03_lora_manager_and_cache.md：
  梳理 LoRA manager 如何加载、缓存、pin、卸载 adapter。

04_worker_model_runner_lora_state.md：
  梳理 Worker / ModelRunner / InputBatch 如何维护当前 batch 的 active LoRA 状态。

05_lora_layer_injection.md：
  梳理 LoRA layer 如何包装 Linear、Embedding、LM head、MoE 并在 forward 中叠加 delta。

06_batch_mixed_lora_execution.md：
  梳理同一 batch 中不同请求使用不同 LoRA 时如何执行。

07_lora_loading_and_weight_mapping.md：
  梳理 LoRA checkpoint 的权重命名、rank、alpha、target modules 和加载映射。

08_lora_and_quantization.md：
  梳理量化 base model 与 LoRA adapter 共存方式和限制。

09_lora_and_parallelism.md：
  梳理 TP / PP / DP、多模态 tower / connector 下 LoRA 权重和 mapping 如何对齐。

10_lora_lifecycle_and_control.md：
  梳理 add_lora、remove_lora、pin_lora、list_loras、shutdown 等控制接口。

11_lora_limitations_and_debugging.md：
  梳理常见限制、错误、fallback 和调试入口。
```

---

## 19. 一句话总结

LoRA 在 vLLM 中是一套横跨请求、调度、Worker 状态、adapter manager、layer wrapper 和 kernel metadata 的动态 adapter 执行系统。

最核心的闭环是：

```text
LoRARequest 选择 adapter
  → SchedulerOutput 把请求状态送到 Worker
  → InputBatch 记录 req_index 到 LoRA id
  → ModelRunner 每轮生成 token 级 LoRAMapping
  → WorkerLoRAManager 加载 / 激活 adapter
  → LoRAModelManager 更新 punica metadata
  → LoRA layer 在 forward 中叠加 delta
```

可以用一句话记住：

```text
LoRARequest 说明“想用谁”，InputBatch 说明“本轮每个 token 用谁”，manager 保证“权重可用”，LoRA layer 完成“把 delta 加上去”。
```
