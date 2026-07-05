# 03. LoRA manager 如何加载、缓存和卸载 adapter？

源码位置：

- `code/vllm/vllm/config/lora.py`
- `code/vllm/vllm/lora/request.py`
- `code/vllm/vllm/lora/worker_manager.py`
- `code/vllm/vllm/lora/model_manager.py`
- `code/vllm/vllm/lora/lora_model.py`
- `code/vllm/vllm/lora/lora_weights.py`
- `code/vllm/vllm/lora/peft_helper.py`
- `code/vllm/vllm/lora/utils.py`
- `code/vllm/vllm/lora/layers/`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/executor/abstract.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`

本问题关注：vLLM V1 中 LoRA adapter 从请求携带 `LoRARequest`，到 Worker 侧加载 checkpoint、注册到 CPU cache、激活到 GPU slot、建立本轮 token 到 adapter 的映射，再到卸载、LRU 淘汰、pin 和查询的完整生命周期。

---

## 1. 一句话回答

LoRA manager 是 Worker / ModelRunner 侧的 adapter 资源管理器。

完整链路可以压缩成：

```text
请求携带 LoRARequest
  → Scheduler 保证本轮不同 LoRA 数不超过 max_loras
  → SchedulerOutput 把 lora_request 传给 Worker
  → InputBatch 记录 req_index -> lora_id
  → GPUModelRunner.set_active_loras()
  → WorkerLoRAManager 加载缺失 adapter
  → LoRAModelManager 注册 LoRAModel 到 CPU cache
  → activate_adapter() 把权重写入 LoRA layer 的 GPU slot
  → set_adapter_mapping() 设置 token/request 到 LoRA slot 的映射
  → forward 时 LoRA layer 按 mapping 执行对应 adapter
```

所以可以这样记：

```text
WorkerLoRAManager 负责“adapter 请求级生命周期”；
LoRAModelManager 负责“模型包装、CPU cache、GPU slots、LoRA layer 权重写入”；
InputBatch / ModelRunner 负责“本轮 batch 哪些 token 使用哪个 LoRA”。
```

---

## 2. 关键对象总览

### 2.1 `LoRARequest`

源码位置：`code/vllm/vllm/lora/request.py:8`

它是一次请求引用 LoRA adapter 的描述对象。

核心字段：

```text
lora_name:
  adapter 名称。

lora_int_id:
  adapter 的整数 ID，必须 > 0。
  vLLM 期望同一个 adapter 使用全局唯一 id，但源码注释说目前没有强制验证全局唯一。

lora_path:
  adapter checkpoint 路径，不能为空。

base_model_name:
  可选的基础模型名。

tensorizer_config_dict:
  可选 tensorizer 加载配置。

load_inplace:
  如果为 True，即使同 id adapter 已经在 cache 中，也强制重新加载并替换。

is_3d_lora_weight:
  MoE LoRA 权重格式标记，配合 enable_mixed_moe_lora_format 使用。
```

位置：`request.py:14` 到 `request.py:37`

它还提供统一 adapter 接口：

```python
adapter_id = lora_int_id
name = lora_name
path = lora_path
```

位置：`request.py:46` 到 `request.py:56`

### 2.2 `WorkerLoRAManager`

源码位置：`code/vllm/vllm/lora/worker_manager.py:25`

它是 Worker / ModelRunner 侧面向请求和控制面的管理器。

职责是：

```text
- create_lora_manager()：创建底层 LoRAModelManager，并把模型包上 LoRA layer；
- add_adapter()：加载并添加 adapter；
- remove_adapter()：移除 adapter；
- pin_adapter()：pin adapter，防止 LRU 淘汰；
- list_adapters()：查询已注册 adapter；
- set_active_adapters()：确保本轮需要的 adapter 已加载，并设置 LoRA mapping；
- add_dummy_lora()：warmup / CUDA graph capture 时创建 dummy LoRA。
```

普通 `WorkerLoRAManager` 的语义比较严格：每次 `_apply_adapters()` 会移除不在本轮请求集合里的 adapter。

位置：`worker_manager.py:194` 到 `worker_manager.py:219`

V1 实际加载模型时使用的是 LRU 版本。

### 2.3 `LRUCacheWorkerLoRAManager`

源码位置：`worker_manager.py:231`

它继承 `WorkerLoRAManager`，底层 manager 换成 `LRUCacheLoRAModelManager`。

它的行为是：

```text
- 本轮请求需要的 adapter 如果已加载，只 touch 一下刷新 LRU 顺序；
- 如果未加载，先从 checkpoint 加载成 LoRAModel；
- 如超过 max_cpu_loras，淘汰最旧 adapter；
- add 后 activate adapter；
- 不会每轮主动卸载非本轮 adapter，而是让 LRU cache 保留它们。
```

对应源码：`worker_manager.py:258` 到 `worker_manager.py:307`

### 2.4 `LoRAModelManager`

源码位置：`code/vllm/vllm/lora/model_manager.py:64`

它是真正和模型结构、LoRA layer、CPU/GPU cache 打交道的对象。

核心字段：

```text
_registered_adapters:
  adapter_id -> LoRAModel
  这是 CPU 侧已注册 adapter cache。

_active_adapters:
  adapter_id -> None
  表示已经被激活进 GPU LoRA slot 的 adapter。

lora_index_to_id:
  list[int | None]
  GPU slot index -> adapter id。
  长度等于 max_loras。

modules:
  module_name -> BaseLayerWithLoRA
  模型中被 LoRA 包装的模块。

packed_modules:
  packed module 的映射，用于 qkv / gate_up 等 packed 权重处理。

punica_wrapper_mapping:
  不同模型前缀对应的 Punica wrapper，用于 language / multimodal tower / connector LoRA kernel metadata。
```

位置：`model_manager.py:88` 到 `model_manager.py:137`

### 2.5 `LoRAModel`

源码位置：`code/vllm/vllm/lora/lora_model.py:60`

它是单个 adapter 加载后的内存表示。

核心字段：

```text
id:
  LoRA adapter int id。

rank:
  LoRA rank。

loras:
  module_name -> LoRALayerWeights
  每个目标模块对应的 LoRA A/B 权重。

is_3d_lora_weight:
  MoE adapter 权重格式标记。
```

位置：`lora_model.py:63` 到 `lora_model.py:89`

---

## 3. 两层容量：CPU cache 和 GPU slots

LoRA 不是只有一个“是否加载”的状态。vLLM 里至少要区分两层：

```text
CPU cache / registered adapters:
  保存已经从 checkpoint 加载、处理过的 LoRAModel。
  容量由 max_cpu_loras 控制。

GPU active slots:
  保存当前可被 forward kernels 使用的 LoRA 权重槽位。
  容量由 max_loras 控制。
```

配置定义在：`code/vllm/vllm/config/lora.py:30`

关键字段：

```text
max_lora_rank:
  最大 LoRA rank，默认 16。

max_loras:
  单个 batch 中最多同时使用多少个 LoRA，也是 GPU slot 数量。
  默认 1。

max_cpu_loras:
  CPU 内存中最多缓存多少个 LoRA。
  如果不设置，默认等于 max_loras。
  必须 >= max_loras。

lora_dtype:
  LoRA 权重 dtype，auto 时跟随 base model dtype。

target_modules:
  限制哪些目标模块启用 LoRA。

enable_tower_connector_lora:
  是否对 multimodal tower / connector 启用 LoRA，实验特性。

enable_mixed_moe_lora_format:
  是否允许 2D / 3D MoE LoRA 格式混用。
```

配置校验：`lora.py:108` 到 `lora.py:125`

`LoRAModelManager` 里对应关系是：

```python
capacity = lora_config.max_cpu_loras
lora_slots = lora_config.max_loras
adapter_slots = lora_slots
```

位置：`model_manager.py:273` 到 `model_manager.py:283`

所以：

```text
max_cpu_loras 控制“最多缓存多少个已加载 adapter”；
max_loras 控制“一轮 forward 最多能同时激活多少个 adapter”。
```

---

## 4. LoRA manager 是如何挂到模型上的

V1 的入口在 `LoRAModelRunnerMixin.load_lora_model()`。

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:31`

流程是：

```python
if not supports_lora(model):
    raise ValueError(...)

self.lora_manager = LRUCacheWorkerLoRAManager(
    vllm_config,
    device,
    model.embedding_modules,
)
return self.lora_manager.create_lora_manager(model, vllm_config)
```

位置：`lora_model_runner_mixin.py:31` 到 `lora_model_runner_mixin.py:46`

`GPUModelRunner.load_model()` 加载 base model 后，如果 `lora_config` 存在，会调用：

```python
self.model = self.load_lora_model(self.model, self.vllm_config, self.device)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5167` 到 `gpu_model_runner.py:5170`

这一步做了两件事：

```text
1. 创建 Worker 侧 lora_manager；
2. 通过 LoRAModelManager 把模型中支持 LoRA 的模块替换成 LoRA wrapper。
```

---

## 5. LoRAModelManager 如何包装模型模块

`LoRAModelManager.__init__()` 会：

```text
1. 找出模型支持的 LoRA modules；
2. 初始化 Punica wrapper；
3. 遍历 model.named_modules()；
4. 匹配支持 LoRA 且符合 target_modules 限制的模块；
5. 把原模块替换成 BaseLayerWithLoRA 子类；
6. 记录到 self.modules；
7. 为 LoRA layer 设置 punica_wrapper；
8. 把 self 挂到 model.lora_manager。
```

入口：`code/vllm/vllm/lora/model_manager.py:67`

### 5.1 支持模块发现

```python
self.supported_lora_modules = get_supported_lora_modules(self.model)
```

位置：`model_manager.py:88` 到 `model_manager.py:92`

如果模型不支持任何 LoRA module，会直接 assert。

### 5.2 Punica wrapper 初始化

入口：`model_manager.py:139`

普通语言模型会创建一个默认 wrapper：

```text
DEFAULT_LANGUAGE_WRAPPER_KEY = "language_model"
```

位置：`model_manager.py:139` 到 `model_manager.py:163`

multimodal 模型会根据 `model.get_mm_mapping()` 创建 language / tower / connector 对应 wrapper。

位置：`model_manager.py:164` 到 `model_manager.py:269`

这就是为什么 multimodal tower connector LoRA 需要特殊 mapping：不同阶段的 token 数和 wrapper 前缀不同。

### 5.3 替换模型模块

核心入口：`model_manager.py:375`

`_create_lora_modules()` 遍历 `model.named_modules()`，对匹配模块调用：

```python
replace_submodule(..., from_layer(...))
```

位置：`model_manager.py:385` 到 `model_manager.py:501`

被替换后的模块必须是 `BaseLayerWithLoRA`，否则根据配置报错或 warning。

位置：`model_manager.py:481` 到 `model_manager.py:497`

最后：

```python
self.register_module(module_name, new_module)
new_module.set_mapping(punica_wrapper)
```

位置：`model_manager.py:497` 到 `model_manager.py:501`

这一步之后，模型 forward 路径中的原始 Linear / Embedding / MoE 等模块已经具备 LoRA 权重槽位和 LoRA mapping 能力。

---

## 6. adapter 是如何从 checkpoint 加载的

请求或控制面调用 `add_lora()` 后，最终会进入 `WorkerLoRAManager._load_adapter()`。

入口：`code/vllm/vllm/lora/worker_manager.py:99`

流程是：

```text
1. 从 LoRAModelManager 读取 supported_lora_modules / packed_modules_mapping；
2. 根据 packed modules 展开 expected_lora_modules；
3. 解析 lora_path 的绝对路径；
4. 读取 PEFT adapter config；
5. 校验 LoRA config 是否符合当前部署限制；
6. 读取模型上的 hf_to_vllm_mapper 和 lora_skip_prefixes；
7. 调用 LoRAModel.from_local_checkpoint()；
8. 标记 is_3d_lora_weight；
9. 返回 LoRAModel。
```

对应源码：`worker_manager.py:99` 到 `worker_manager.py:162`

### 6.1 PEFT 配置校验

```python
peft_helper = PEFTHelper.from_local_dir(...)
peft_helper.validate_legal(self.lora_config)
```

位置：`worker_manager.py:114` 到 `worker_manager.py:122`

这一步会在真正加载权重前确认 rank、target modules 等配置是否合法。

### 6.2 checkpoint 文件格式

`LoRAModel.from_local_checkpoint()` 支持：

```text
adapter_model.safetensors
adapter_model.bin
adapter_model.pt
tensorizer adapter_model.tensors
```

入口：`code/vllm/vllm/lora/lora_model.py:167`

文件路径判断：`lora_model.py:205` 到 `lora_model.py:295`

### 6.3 unexpected modules 校验

加载 safetensors / bin / pt 时会检查 checkpoint 中的 LoRA module 是否在 expected modules 内。

位置：`lora_model.py:212` 到 `lora_model.py:242`

如果 checkpoint 里有当前模型不期望的 LoRA module，会抛出 `ValueError`。

### 6.4 权重转换成 `LoRAModel`

`from_lora_tensors()` 会把 tensor name 解析成 module name 和 A/B 权重，生成：

```text
module_name -> LoRALayerWeights
```

入口：`lora_model.py:117`

它会跳过 base embedding 权重，处理 model-defined skip prefixes，并检查 embedding LoRA vocab size 是否和 base model 一致。

位置：`lora_model.py:131` 到 `lora_model.py:164`

---

## 7. add_adapter 的生命周期

V1 默认使用 `LRUCacheWorkerLoRAManager.add_adapter()`。

入口：`code/vllm/vllm/lora/worker_manager.py:273`

核心逻辑：

```text
if adapter 不在 list_adapters() 中，或者 load_inplace=True:
    1. 先从 checkpoint 加载 LoRAModel；
    2. 如果已有同 id adapter，先 remove；
    3. 如果注册数量 + 1 > max_cpu_loras，淘汰最旧 adapter；
    4. add_adapter(lora) 注册到 LoRAModelManager；
else:
    1. adapter 已加载，只 get_adapter() touch LRU 顺序；

最后：
    activate_adapter(lora_int_id)
```

位置：`worker_manager.py:273` 到 `worker_manager.py:307`

这里有一个重要设计：

```text
先加载新 adapter，确认 checkpoint 有效，再淘汰旧 adapter。
```

源码注释说明这会让已加载 LoRA 数短暂超过 `max_cpu_loras`，但可以避免因为新 adapter 无效而先把旧 adapter 淘汰掉。

位置：`worker_manager.py:279` 到 `worker_manager.py:287`

### 7.1 注册到 CPU cache

`LRUCacheLoRAModelManager.add_adapter()` 会：

```text
- 如果 adapter id 不在 _registered_adapters：调用 _add_adapter(lora)；
- 如果已经存在：touch LRU 顺序；
```

位置：`model_manager.py:1196` 到 `model_manager.py:1206`

`_add_adapter()` 会：

```python
self._create_merged_loras_inplace(lora)
self._registered_adapters[lora.id] = lora
```

位置：`model_manager.py:333` 到 `model_manager.py:335`

### 7.2 合并 packed LoRA 权重

`_create_merged_loras_inplace()` 会处理 qkv / gate_up / MoE experts 等 packed module，把多个 checkpoint module 的 LoRA 权重合并成 LoRA layer 实际需要的格式。

入口：`model_manager.py:724`

它还会：

```text
- 处理 pooling model 的 model. 前缀差异；
- 处理 FusedMoE / FusedMoE3D；
- expert parallel 下裁剪本 rank local experts；
- mixed MoE format 下做 3D -> 2D 转换；
- 对 LoRA weights 做 optimize()；
- 在 CPU 且平台支持时 pin_memory。
```

位置：`model_manager.py:724` 到 `model_manager.py:815`

### 7.3 激活到 GPU slot

注册完成后会调用：

```python
self._adapter_manager.activate_adapter(lora_request.lora_int_id)
```

位置：`worker_manager.py:306`

`LoRAModelManager.activate_adapter()` 会：

```text
1. 如果 adapter 已经 active，直接返回 False；
2. 找到第一个空 GPU slot；
3. _active_adapters[lora_id] = None；
4. lora_index_to_id[slot] = lora_id；
5. 遍历所有 LoRA wrapped modules；
6. 取该 adapter 在该 module 上的 LoRA weights；
7. module.set_lora(slot, lora_a, lora_b) 写入 LoRA layer 的 slot；
8. 如果该 module 没有 LoRA weights，module.reset_lora(slot)。
```

位置：`model_manager.py:285` 到 `model_manager.py:324`

这一步才是真正让 adapter 可被 forward kernels 使用。

---

## 8. LRU cache 和淘汰策略

### 8.1 两个 LRU cache

`LRUCacheLoRAModelManager` 初始化时会把两个 dict 换成 LRU cache：

```python
self._registered_adapters = LoRALRUCache(self.capacity, self.deactivate_adapter)
self._active_adapters = LoRALRUCache(self.lora_slots, self._deactivate_adapter)
```

位置：`model_manager.py:1185` 到 `model_manager.py:1190`

含义：

```text
_registered_adapters:
  CPU cache，容量 max_cpu_loras。
  移除时会调用 deactivate_adapter()，确保 GPU slot 也不再 active。

_active_adapters:
  GPU slot LRU，容量 max_loras。
  移除时调用 _deactivate_adapter()，只清 slot 映射。
```

### 8.2 LRU 移除时会 deactivate

`AdapterLRUCache._on_remove()` 会调用传入的 `deactivate_fn`。

位置：`model_manager.py:53` 到 `model_manager.py:61`

所以 CPU cache 淘汰 adapter 时，会先从 active 状态中移除；GPU active LRU 淘汰 adapter 时，会清掉对应 slot。

### 8.3 CPU cache 淘汰

`LRUCacheWorkerLoRAManager.add_adapter()` 中：

```python
if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
    self._adapter_manager.remove_oldest_adapter()
```

位置：`worker_manager.py:293` 到 `worker_manager.py:297`

`remove_oldest_adapter()` 会移除 registered adapters 中最旧的条目。

位置：`model_manager.py:1222` 到 `model_manager.py:1226`

### 8.4 GPU active slot 淘汰

`LRUCacheLoRAModelManager.activate_adapter()` 中：

```python
if lora_id not in _active_adapters and len(_active_adapters) >= lora_slots:
    _active_adapters.remove_oldest()
```

位置：`model_manager.py:1208` 到 `model_manager.py:1220`

然后再调用父类 `activate_adapter()` 把当前 adapter 写入空出来的 slot。

### 8.5 touch 行为

如果 adapter 已注册但再次被请求：

```python
loaded = self._adapter_manager.get_adapter(lora_request.lora_int_id) is not None
```

位置：`worker_manager.py:300` 到 `worker_manager.py:305`

对 LRU manager 来说，`get_adapter()` 访问 registered cache 会刷新 LRU 顺序；`activate_adapter()` 后也会 touch active cache。

位置：`model_manager.py:1217` 到 `model_manager.py:1219`

---

## 9. pin_lora 如何工作

V1 控制面调用最终会进入：

```text
Executor.pin_lora()
  → Worker.pin_lora()
  → ModelRunner.pin_lora()
  → WorkerLoRAManager.pin_adapter()
  → LRUCacheLoRAModelManager.pin_adapter()
```

入口：

- `code/vllm/vllm/v1/executor/abstract.py:300`
- `code/vllm/vllm/v1/worker/gpu_worker.py:967`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:282`
- `code/vllm/vllm/lora/worker_manager.py:180`
- `code/vllm/vllm/lora/model_manager.py:1228`

`LRUCacheLoRAModelManager.pin_adapter()` 会同时 pin 两层：

```python
self._pin_lora_in_cpu_cache(lora_id)
self._pin_lora_in_gpu_cache(lora_id)
```

位置：`model_manager.py:1228` 到 `model_manager.py:1232`

CPU cache pin：

```text
_registered_adapters.pin(lora_id)
```

如果 adapter 未注册，会抛出：

```text
Pinning failed. LoRA {lora_id} is not registered.
```

位置：`model_manager.py:1234` 到 `model_manager.py:1240`

GPU cache pin：

```text
如果 adapter 不在 active adapters，先 activate_adapter(lora_id)；
然后 _active_adapters.pin(lora_id)。
```

位置：`model_manager.py:1242` 到 `model_manager.py:1247`

所以 pin 的语义是：

```text
adapter 不仅留在 CPU cache，还会被激活进 GPU slot，并防止被 LRU 淘汰。
```

前提是 active slot 容量允许，否则 activate 过程可能因为没有可用 slot 或 pinned 太多而失败。

---

## 10. remove_lora 和 list_loras

### 10.1 remove_lora

控制面路径：

```text
Executor.remove_lora(lora_id)
  → collective_rpc("remove_lora")
  → Worker.remove_lora(lora_id)
  → ModelRunner.remove_lora(lora_id)
  → WorkerLoRAManager.remove_adapter(lora_id)
  → LoRAModelManager.remove_adapter(lora_id)
```

`LoRAModelManager.remove_adapter()` 会：

```text
1. deactivate_adapter(adapter_id)；
2. 如果 adapter 不在 _registered_adapters，返回 False；
3. 从 _registered_adapters 删除 adapter；
4. 返回 True。
```

位置：`model_manager.py:1144` 到 `model_manager.py:1149`

`deactivate_adapter()` 会：

```text
- 如果 adapter 不 active，返回 False；
- 清 lora_index_to_id 中的 slot；
- 从 _active_adapters 移除。
```

位置：`model_manager.py:1123` 到 `model_manager.py:1128`

### 10.2 list_loras

控制面路径：

```text
Executor.list_loras()
  → collective_rpc("list_loras")
  → 每个 Worker 返回 set[int]
  → Executor 断言所有 worker 的集合一致
  → 返回第一个集合
```

位置：`code/vllm/vllm/v1/executor/abstract.py:304` 到 `abstract.py:308`

Worker / ModelRunner / Manager 最终返回 registered adapters 的 id 集合：

```python
return set(self._adapter_manager.list_adapters())
```

位置：`worker_manager.py:227` 到 `worker_manager.py:228`

对于 LRU manager，`list_adapters()` 返回 `_registered_adapters.cache` 的拷贝。

位置：`model_manager.py:1192` 到 `model_manager.py:1194`

---

## 11. 多 worker 一致性由谁保证

Executor 层的 LoRA 控制面都是 collective RPC。

位置：`code/vllm/vllm/v1/executor/abstract.py:292` 到 `abstract.py:308`

```python
add_lora(...):
  return all(self.collective_rpc("add_lora", args=(lora_request,)))

remove_lora(...):
  return all(self.collective_rpc("remove_lora", args=(lora_id,)))

pin_lora(...):
  return all(self.collective_rpc("pin_lora", args=(lora_id,)))

list_loras():
  sets = self.collective_rpc("list_loras")
  assert s == sets[0] for every worker
```

这意味着：

```text
- add/remove/pin 会广播到所有 worker；
- 返回值必须所有 worker 都成功才算成功；
- list_loras 会检查所有 worker 的 adapter 集合一致；
- 如果某个 worker 加载失败，整体 add_lora 返回 False 或抛错。
```

---

## 12. Scheduler 如何约束本轮 LoRA 数量

LoRA manager 虽然有 `max_loras` GPU slot 限制，但 Scheduler 会提前避免同一轮调度超过这个限制。

在 `Scheduler.schedule()` 中，running 请求会先收集本轮已调度请求的 LoRA id：

```python
scheduled_loras = set(
    req.lora_request.lora_int_id
    for req in scheduled_running_reqs
    if req.lora_request and req.lora_request.lora_int_id > 0
)
assert len(scheduled_loras) <= self.lora_config.max_loras
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:614` 到 `scheduler.py:622`

调度 waiting 请求时，如果加入新请求会让本轮不同 LoRA 数超过 `max_loras`，就跳过该请求：

```text
len(scheduled_loras) == max_loras
and request.lora_request.lora_int_id not in scheduled_loras
```

位置：`scheduler.py:651` 到 `scheduler.py:664`

所以：

```text
Scheduler 控制“这一轮最多调度多少种 LoRA”；
LoRAModelManager 控制“当前 worker cache 里最多注册/激活多少 LoRA”。
```

---

## 13. InputBatch 如何保存请求到 LoRA 的映射

`InputBatch` 是执行期把 request state 变成 tensor / mapping 的地方。

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py`

初始化时有三个 LoRA 相关字段：

```python
self.request_lora_mapping = np.zeros((self.max_num_reqs,), dtype=np.int64)
self.lora_id_to_request_ids: dict[int, set[str]] = {}
self.lora_id_to_lora_request: dict[int, LoRARequest] = {}
```

位置：`gpu_input_batch.py:244` 到 `gpu_input_batch.py:247`

### 13.1 add_request

当新请求进入 batch，如果它有 `lora_request`：

```text
request_lora_mapping[req_index] = lora_id
lora_id_to_request_ids[lora_id].add(req_id)
lora_id_to_lora_request[lora_id] = request.lora_request
```

位置：`gpu_input_batch.py:468` 到 `gpu_input_batch.py:479`

如果请求没有 LoRA：

```text
request_lora_mapping[req_index] = 0
```

这里的 `0` 表示无 LoRA。

### 13.2 remove_request

请求从 batch 移除时：

```text
- 找到 req_index 对应 lora_id；
- 从 lora_id_to_request_ids[lora_id] 移除 req_id；
- 如果这个 lora_id 已经没有请求引用，从 batch 的两个映射字典删除；
- request_lora_mapping[req_index] = 0。
```

位置：`gpu_input_batch.py:530` 到 `gpu_input_batch.py:538`

注意：

```text
这只清理 batch 内部的引用，
不会直接从 LoRAModelManager 的 CPU cache 或 GPU active slots 删除 adapter。
```

adapter 是否继续留在 cache，取决于 LRU / remove_lora / pin_lora。

### 13.3 make_lora_inputs

执行前，ModelRunner 调用：

```python
input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)
```

入口：`gpu_input_batch.py:976`

返回三个对象：

```text
prompt_lora_mapping:
  size = sum(num_sampled_tokens)
  第 i 个 sampled token 使用哪个 LoRA id。

token_lora_mapping:
  size = sum(num_scheduled_tokens)
  第 i 个 scheduled input token 使用哪个 LoRA id。

active_lora_requests:
  当前 batch 中相关 LoRARequest 集合。
```

位置：`gpu_input_batch.py:976` 到 `gpu_input_batch.py:999`

这三个对象会进入 `LoRAMapping`。

---

## 14. ModelRunner 如何在执行前激活本轮 LoRA

`GPUModelRunner` 在准备输入阶段会设置 active LoRA。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2193` 到 `gpu_model_runner.py:2201`

```python
if self.lora_config:
    self.set_active_loras(
        self.input_batch, num_scheduled_tokens, num_sampled_tokens
    )
```

`set_active_loras()` 定义在 mixin：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:73`

流程是：

```text
1. InputBatch.make_lora_inputs() 生成 prompt_lora_mapping / token_lora_mapping / lora_requests；
2. 构造 LoRAMapping；
3. WorkerLoRAManager.set_active_adapters(lora_requests, lora_mapping)；
4. manager 确保这些 adapter 已加载并 active；
5. LoRAModelManager.set_adapter_mapping() 更新 Punica wrapper metadata。
```

关键代码：`lora_model_runner_mixin.py:73` 到 `lora_model_runner_mixin.py:91`

`_set_active_loras()` 中：

```python
lora_mapping = LoRAMapping(
    token_lora_mapping,
    prompt_lora_mapping,
    is_prefill=True,
    type=mapping_type,
)
self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
```

位置：`lora_model_runner_mixin.py:48` 到 `lora_model_runner_mixin.py:67`

`WorkerLoRAManager.set_active_adapters()` 会：

```python
self._apply_adapters(requests)
self._adapter_manager.set_adapter_mapping(mapping)
```

位置：`code/vllm/vllm/lora/worker_manager.py:183` 到 `worker_manager.py:186`

---

## 15. LoRA mapping 如何进入 kernel metadata

`LoRAModelManager.set_adapter_mapping()` 会在 mapping 变化时调用 `_set_adapter_mapping()`。

位置：`code/vllm/vllm/lora/model_manager.py:1139` 到 `model_manager.py:1142`

`_set_adapter_mapping()` 会根据 mapping 类型选择对应 wrapper：

```text
LANGUAGE:
  language model wrapper。

TOWER:
  multimodal tower wrapper。

CONNECTOR:
  multimodal connector wrapper。
```

位置：`model_manager.py:344` 到 `model_manager.py:367`

最后调用：

```python
punica_wrapper.update_metadata(
    mapping,
    self.lora_index_to_id,
    self.lora_slots + 1,
    self.vocab_size,
)
```

位置：`model_manager.py:362` 到 `model_manager.py:367`

这一步把两个信息交给 LoRA kernel wrapper：

```text
1. 每个 token / sampled token 对应哪个 LoRA id；
2. 当前 LoRA id 在 GPU slot 中的 index 是什么。
```

因此 forward 时 LoRA layer 可以根据 token mapping 选择正确 adapter slot。

---

## 16. multimodal tower / connector LoRA 特殊路径

普通 language LoRA 在 `_prepare_inputs()` 里通过 `set_active_loras()` 设置。

multimodal tower / connector LoRA 发生在 `_execute_mm_encoder()` 中，因为 encoder batch 的组织和 language token batch 不一样。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2941` 到 `gpu_model_runner.py:3008`

触发条件：

```python
if self.lora_config and self.lora_manager.supports_tower_connector_lora():
```

位置：`gpu_model_runner.py:2941`

流程是：

```text
1. 遍历本轮 scheduled encoder inputs；
2. 根据 req_id 找 input_batch 中的 lora_id；
3. 为 tower encoder tokens 构造 token_lora_mapping / prompt_lora_mapping；
4. 收集相关 lora_requests；
5. 构造 LoRAMapping(type=TOWER)；
6. lora_manager.set_active_adapters(..., tower_mapping)；
7. 如果模型有 connector，再构造 LoRAMapping(type=CONNECTOR) 并设置。
```

关键位置：

- tower mapping：`gpu_model_runner.py:2941` 到 `gpu_model_runner.py:2973`
- connector mapping：`gpu_model_runner.py:2975` 到 `gpu_model_runner.py:3008`

这说明：

```text
同一个 LoRA adapter 可能需要分别给 language model、multimodal tower、connector 设置不同的 mapping，
因为它们的 token 数、batch 结构和 wrapper 前缀不同。
```

---

## 17. dummy LoRA 和 warmup / CUDA graph capture

LoRA 开启后，warmup / CUDA graph capture 需要覆盖有 LoRA 的执行路径。

`LoRAModelRunnerMixin` 提供三个 context manager：

```text
maybe_setup_dummy_loras()
maybe_select_dummy_loras()
maybe_dummy_run_with_lora()
```

位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:93` 到 `lora_model_runner_mixin.py:272`

### 17.1 dummy LoRA 创建

`maybe_setup_dummy_loras()` 会：

```text
1. 根据 max_loras 创建 lora_id = 1..max_loras 的 dummy LoRARequest；
2. 进入 dummy_lora_cache()，复用 dummy LoRAModel，避免重复创建；
3. 调用 add_dummy_lora()；
4. context 退出时根据 remove_lora 决定是否 remove_all_adapters()。
```

位置：`lora_model_runner_mixin.py:93` 到 `lora_model_runner_mixin.py:131`

`WorkerLoRAManager.add_dummy_lora()` 会调用底层 manager 的 `create_dummy_lora()`。

位置：`code/vllm/vllm/lora/worker_manager.py:164` 到 `worker_manager.py:175`

### 17.2 dummy rank

默认 warmup rank 是：

```text
min(max_lora_rank, 8)
```

然后会调用 `get_dummy_lora_warmup_rank()`，在 fully sharded MoE 场景下调整到 tensor parallel size 的倍数。

位置：

- `lora_model_runner_mixin.py:103` 到 `lora_model_runner_mixin.py:109`
- `model_manager.py:640` 到 `model_manager.py:670`

### 17.3 dummy mapping

`maybe_select_dummy_loras()` 会构造模拟的 prompt / token LoRA mapping，用于 capture 不同 active LoRA 数量的图。

位置：`lora_model_runner_mixin.py:132` 到 `lora_model_runner_mixin.py:235`

这也是为什么 LoRA 会影响 CUDA graph specialization：active LoRA 数量会进入 dispatch / capture key。

`GPUModelRunner` 中计算 active LoRA 数：

```python
num_active_loras = len(self.input_batch.lora_id_to_lora_request)
has_lora = num_active_loras > 0
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3845` 到 `gpu_model_runner.py:3851`

---

## 18. 控制面 add/remove/pin/list 的完整路径

### 18.1 add_lora

```text
Engine / API 层
  → Executor.add_lora(lora_request)
  → collective_rpc("add_lora")
  → Worker.add_lora(lora_request)
  → ModelRunner.add_lora(lora_request)
  → WorkerLoRAManager.add_adapter(lora_request)
  → _load_adapter()
  → LoRAModel.from_local_checkpoint()
  → LRUCacheLoRAModelManager.add_adapter()
  → activate_adapter()
```

关键源码：

- Executor：`abstract.py:292` 到 `abstract.py:294`
- Worker：`gpu_worker.py:958` 到 `gpu_worker.py:959`
- ModelRunner mixin：`lora_model_runner_mixin.py:274` 到 `lora_model_runner_mixin.py:276`
- Worker manager：`worker_manager.py:273` 到 `worker_manager.py:307`

### 18.2 remove_lora

```text
Executor.remove_lora(lora_id)
  → collective_rpc("remove_lora")
  → Worker.remove_lora(lora_id)
  → ModelRunner.remove_lora(lora_id)
  → WorkerLoRAManager.remove_adapter(lora_id)
  → LoRAModelManager.remove_adapter(lora_id)
  → deactivate_adapter(lora_id)
  → 从 registered cache 删除
```

关键源码：

- Executor：`abstract.py:296` 到 `abstract.py:298`
- Worker：`gpu_worker.py:961` 到 `gpu_worker.py:962`
- ModelRunner mixin：`lora_model_runner_mixin.py:278` 到 `lora_model_runner_mixin.py:280`
- Model manager：`model_manager.py:1123` 到 `model_manager.py:1149`

### 18.3 pin_lora

```text
Executor.pin_lora(lora_id)
  → collective_rpc("pin_lora")
  → Worker.pin_lora(lora_id)
  → ModelRunner.pin_lora(lora_id)
  → WorkerLoRAManager.pin_adapter(lora_id)
  → LRUCacheLoRAModelManager.pin_adapter(lora_id)
  → pin CPU cache + pin GPU active cache
```

关键源码：

- Executor：`abstract.py:300` 到 `abstract.py:302`
- Worker：`gpu_worker.py:967` 到 `gpu_worker.py:968`
- ModelRunner mixin：`lora_model_runner_mixin.py:282` 到 `lora_model_runner_mixin.py:284`
- Model manager：`model_manager.py:1228` 到 `model_manager.py:1247`

### 18.4 list_loras

```text
Executor.list_loras()
  → collective_rpc("list_loras")
  → Worker.list_loras()
  → ModelRunner.list_loras()
  → WorkerLoRAManager.list_adapters()
  → 返回 registered adapters 的 id 集合
  → Executor 断言所有 worker 集合一致
```

关键源码：

- Executor：`abstract.py:304` 到 `abstract.py:308`
- Worker：`gpu_worker.py:964` 到 `gpu_worker.py:965`
- ModelRunner mixin：`lora_model_runner_mixin.py:286` 到 `lora_model_runner_mixin.py:288`
- Worker manager：`worker_manager.py:227` 到 `worker_manager.py:228`

---

## 19. 请求执行中的完整例子

假设请求 `R1` 携带：

```text
LoRARequest(
  lora_name="sql_adapter",
  lora_int_id=7,
  lora_path="/models/lora/sql_adapter"
)
```

### 19.1 请求进入调度

```text
Request.lora_request = LoRARequest(...)
Scheduler.schedule()
  → scheduled_loras 加入 7
  → 如果本轮不同 LoRA 数未超过 max_loras，R1 可以被调度
  → NewRequestData.from_request() 把 lora_request 发给 Worker
```

`NewRequestData` 字段位置：`code/vllm/vllm/v1/core/sched/output.py:31` 到 `output.py:65`

### 19.2 Worker 建立 batch 状态

`GPUModelRunner._update_states()` 创建 `CachedRequestState` 时保存 `lora_request`。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1232` 到 `gpu_model_runner.py:1237`

`InputBatch.add_request()` 记录：

```text
request_lora_mapping[req_index] = 7
lora_id_to_request_ids[7].add(R1)
lora_id_to_lora_request[7] = LoRARequest(...)
```

位置：`gpu_input_batch.py:468` 到 `gpu_input_batch.py:477`

### 19.3 执行前激活 LoRA

`GPUModelRunner` 准备输入时：

```text
InputBatch.make_lora_inputs()
  → token_lora_mapping 中 R1 的 scheduled tokens 都映射到 7
  → prompt_lora_mapping 中 R1 的 sampled token 映射到 7
  → active_lora_requests = {LoRARequest(...)}

LoRAModelRunnerMixin._set_active_loras()
  → LoRAMapping(...)
  → lora_manager.set_active_adapters(...)
```

位置：

- `gpu_input_batch.py:976` 到 `gpu_input_batch.py:999`
- `lora_model_runner_mixin.py:48` 到 `lora_model_runner_mixin.py:67`

### 19.4 adapter 未加载时

```text
LRUCacheWorkerLoRAManager.add_adapter(LoRARequest id=7)
  → _load_adapter()
  → LoRAModel.from_local_checkpoint()
  → _adapter_manager.add_adapter(lora)
  → _adapter_manager.activate_adapter(7)
```

激活后：

```text
lora_index_to_id[slot] = 7
每个 BaseLayerWithLoRA 的 slot 上写入 sql_adapter 的 LoRA A/B 权重
punica_wrapper metadata 记录 token -> lora_id -> slot 的映射
```

### 19.5 adapter 已加载时

如果 id=7 已经在 registered cache：

```text
不重新读 checkpoint；
刷新 LRU 顺序；
确保 id=7 active；
更新本轮 LoRAMapping。
```

这就是 LoRA cache 命中的路径。

### 19.6 请求结束后

请求从 `InputBatch` 移除时：

```text
InputBatch.remove_request(R1)
  → 从 lora_id_to_request_ids[7] 删除 R1
  → 如果没有其他 request 使用 7，从 batch 引用字典删除 7
```

位置：`gpu_input_batch.py:530` 到 `gpu_input_batch.py:538`

但 adapter 7 不会因此立刻从 LoRAModelManager 删除。

```text
它仍可能留在 CPU cache / GPU active cache，等待后续请求复用；
只有 remove_lora()、remove_all_adapters() 或 LRU 淘汰才会真正删除。
```

---

## 20. 常见问题

### 20.1 `max_loras` 和 `max_cpu_loras` 有什么区别？

```text
max_loras:
  单 batch 最多同时使用的 LoRA 数量，也是 GPU active slot 数。

max_cpu_loras:
  Worker CPU 内存中最多缓存的已加载 LoRAModel 数。
```

如果 `max_cpu_loras` 不设置，默认等于 `max_loras`。

位置：`config/lora.py:108` 到 `config/lora.py:116`

### 20.2 adapter 加载后一定 active 吗？

通过 `add_adapter()` 加载的 adapter 会立即 `activate_adapter()`。

位置：`worker_manager.py:306`

但后续它可能因为 GPU active LRU 容量不足被 deactive；CPU registered cache 中仍可能保留。

### 20.3 active adapter 和 registered adapter 的区别是什么？

```text
registered:
  LoRAModel 已经在 CPU cache 中，说明 checkpoint 已加载并处理过。

active:
  LoRA 权重已经写入某个 GPU slot，forward 可以通过 LoRA mapping 使用它。
```

同一个 adapter 可以 registered 但不 active。

### 20.4 为什么 Scheduler 还要限制 `max_loras`？

因为一轮 batch 中如果不同 LoRA 数超过 GPU slots，执行层无法同时给所有 token 设置有效映射。

Scheduler 在加入 waiting 请求前就跳过会超过 `max_loras` 的请求。

位置：`scheduler.py:651` 到 `scheduler.py:664`

### 20.5 `lora_int_id=0` 表示什么？

`LoRARequest.__post_init__()` 要求 id > 0。

位置：`request.py:39` 到 `request.py:45`

执行期 `InputBatch.request_lora_mapping` 中的 `0` 表示这个 request/token 不使用 LoRA。

位置：`gpu_input_batch.py:477` 到 `gpu_input_batch.py:479`

### 20.6 `load_inplace=True` 用来做什么？

如果同一个 `lora_int_id` 已经在 cache 中，默认不会重新加载。

`load_inplace=True` 会强制重新从路径加载 adapter，并先移除旧 adapter，再注册新 adapter。

位置：`request.py:19` 到 `request.py:22`，`worker_manager.py:279` 到 `worker_manager.py:292`

### 20.7 pin 后还能 remove_lora 吗？

pin 的目的是防止 LRU 淘汰，不是禁止显式删除。

显式 `remove_lora(lora_id)` 仍会调用 `LoRAModelManager.remove_adapter()`，先 deactivate 再从 registered cache 删除。

位置：`model_manager.py:1144` 到 `model_manager.py:1149`

### 20.8 list_loras 返回 active 还是 registered？

返回 registered adapters，也就是 CPU cache 中已加载的 adapter id 集合。

位置：`worker_manager.py:227` 到 `worker_manager.py:228`，`model_manager.py:1192` 到 `model_manager.py:1194`

### 20.9 为什么 add_lora 要广播到所有 worker？

因为 tensor parallel / pipeline parallel / data parallel worker 都可能参与 forward，每个 worker 都需要持有与自己 rank 对应的 LoRA 权重和 slot 状态。

Executor 用 `collective_rpc("add_lora")` 广播，并用 `all()` 汇总成功状态。

位置：`abstract.py:292` 到 `abstract.py:294`

### 20.10 adapter path 加载失败会怎样？

如果路径找不到 adapter，会从 `_load_adapter()` 抛出 `LoRAAdapterNotFoundError`。

位置：`worker_manager.py:150` 到 `worker_manager.py:158`

如果 checkpoint module 不符合当前模型 expected modules，会在 `LoRAModel.from_local_checkpoint()` 中抛 `ValueError`。

位置：`lora_model.py:236` 到 `lora_model.py:242`

---

## 21. 生命周期总览图

```text
初始化
  → GPUModelRunner.load_model()
  → load_lora_model()
  → LRUCacheWorkerLoRAManager
  → LoRAModelManager
  → 替换模型中的 LoRA target modules
  → 创建 CPU cache / GPU slots / Punica wrappers

控制面加载
  → Executor.add_lora()
  → Worker / ModelRunner add_lora()
  → WorkerLoRAManager._load_adapter()
  → PEFT config 校验
  → LoRAModel.from_local_checkpoint()
  → LoRAModelManager.add_adapter()
  → registered CPU cache
  → activate_adapter()
  → 写入 GPU LoRA slot

请求执行
  → Scheduler 限制本轮 LoRA 数 <= max_loras
  → SchedulerOutput 携带 lora_request
  → InputBatch 记录 req_index -> lora_id
  → make_lora_inputs()
  → LoRAMapping(token_lora_mapping, prompt_lora_mapping)
  → set_active_adapters()
  → 缺失 adapter 加载 / 已有 adapter touch
  → set_adapter_mapping()
  → forward kernels 根据 mapping 使用 LoRA slot

回收和缓存
  → request 从 InputBatch 移除，只清 batch 引用
  → adapter 继续留在 registered / active cache
  → CPU cache 超过 max_cpu_loras 时淘汰 oldest registered adapter
  → GPU active 超过 max_loras 时淘汰 oldest active adapter
  → pin_lora 可 pin CPU + GPU cache
  → remove_lora 显式删除 adapter
```

---

## 22. 一句话总结

LoRA manager 的核心不是简单“加载一个 adapter 文件”，而是一个两层缓存和一层映射协议：`LoRAModelManager` 把 checkpoint 变成 CPU cache 中的 `LoRAModel`，再把需要执行的 adapter 激活进有限的 GPU slots；`InputBatch` 和 `LoRAMapping` 告诉 LoRA kernels 每个 token 使用哪个 adapter；LRU、pin、remove 和 Scheduler 的 `max_loras` 约束共同保证 adapter 能被复用但不会超过执行容量。
