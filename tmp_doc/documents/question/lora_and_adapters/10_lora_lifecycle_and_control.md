# 10. LoRA 生命周期和控制接口如何工作？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\llm_engine.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\async_llm.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\core_client.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\core.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\executor\abstract.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\worker_base.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_worker.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\lora_model_runner_mixin.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\worker_manager.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\model_manager.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\lora_model.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\request.py`

本问题关注：`add_lora`、`remove_lora`、`pin_lora`、`list_loras` 等控制接口如何从 Engine / AsyncEngine / CoreClient 传到 Executor 和 Worker，Worker 如何通过 LoRA manager 加载、卸载、pin、查询 adapter，以及这些控制接口与每轮请求调度、active mapping、LRU 缓存之间的边界。

---

## 1. 一句话回答

LoRA 生命周期由执行层控制面管理。

控制请求会沿着：

```text
LLMEngine / AsyncLLM / EngineCoreClient
  → EngineCore
  → model_executor
  → Executor.collective_rpc(...)
  → Worker.add_lora / remove_lora / pin_lora / list_loras
  → ModelRunner LoRA mixin
  → WorkerLoRAManager / LRUCacheWorkerLoRAManager
  → LoRAModelManager / LRUCacheLoRAModelManager
```

最终在 Worker 侧完成：

```text
add_lora：读取 adapter checkpoint，注册到 CPU / manager cache，并激活到 GPU LoRA slot；
remove_lora：deactivate GPU slot，并从 registered adapter cache 删除；
pin_lora：把 adapter 固定在 CPU cache 和 GPU active cache 中，避免 LRU 淘汰；
list_loras：返回当前所有 worker 一致的已注册 LoRA id 集合。
```

关键边界：

```text
请求里的 LoRARequest 是数据面，表示“这个请求要用哪个 adapter”；
add/remove/pin/list 是控制面，表示“worker 当前加载和管理哪些 adapter”。
```

---

## 2. 控制接口总览

V1 同步入口在 `LLMEngine`：`llm_engine.py:403` 到 `llm_engine.py:417`

```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    """Load a new LoRA adapter into the engine for future requests."""
    return self.engine_core.add_lora(lora_request)

def remove_lora(self, lora_id: int) -> bool:
    """Remove an already loaded LoRA adapter."""
    return self.engine_core.remove_lora(lora_id)

def list_loras(self) -> set[int]:
    """List all registered adapters."""
    return self.engine_core.list_loras()

def pin_lora(self, lora_id: int) -> bool:
    """Prevent an adapter from being evicted."""
    return self.engine_core.pin_lora(lora_id)
```

V1 异步入口在 `AsyncLLM`：`async_llm.py:948` 到 `async_llm.py:962`

```python
async def add_lora(self, lora_request: LoRARequest) -> bool:
    return await self.engine_core.add_lora_async(lora_request)

async def remove_lora(self, lora_id: int) -> bool:
    return await self.engine_core.remove_lora_async(lora_id)

async def list_loras(self) -> set[int]:
    return await self.engine_core.list_loras_async()

async def pin_lora(self, lora_id: int) -> bool:
    return await self.engine_core.pin_lora_async(lora_id)
```

如果 EngineCore 在独立进程中运行，`CoreClient` 通过 utility call 转发：

```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    return self.call_utility("add_lora", lora_request)

def remove_lora(self, lora_id: int) -> bool:
    return self.call_utility("remove_lora", lora_id)

def list_loras(self) -> set[int]:
    return self.call_utility("list_loras")

def pin_lora(self, lora_id: int) -> bool:
    return self.call_utility("pin_lora", lora_id)
```

位置：`core_client.py:911` 到 `core_client.py:921`

---

## 3. 主链路

完整控制面链路是：

```text
LLMEngine.add_lora(lora_request)
  → EngineCore.add_lora(lora_request)
  → Executor.add_lora(lora_request)
  → Executor.collective_rpc("add_lora", args=(lora_request,))
  → GPUWorker.add_lora(lora_request)
  → GPUModelRunner.add_lora(lora_request)
  → LoRAModelRunnerMixin.add_lora()
  → WorkerLoRAManager.add_adapter(lora_request)
  → LRUCacheWorkerLoRAManager.add_adapter()
  → WorkerLoRAManager._load_adapter()
  → LoRAModel.from_local_checkpoint()
  → LoRAModelManager.add_adapter()
  → LoRAModelManager.activate_adapter()
  → BaseLayerWithLoRA.set_lora(slot_index, lora_a, lora_b)
```

remove / pin / list 类似，只是落到不同方法：

```text
remove_lora
  → lora_manager.remove_adapter(lora_id)
  → LoRAModelManager.remove_adapter(lora_id)

pin_lora
  → lora_manager.pin_adapter(lora_id)
  → LRUCacheLoRAModelManager.pin_adapter(lora_id)

list_loras
  → lora_manager.list_adapters()
```

---

## 4. EngineCore 到 Executor

`EngineCore` 中的控制接口很薄：`core.py:822` 到 `core.py:832`

```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    return self.model_executor.add_lora(lora_request)

def remove_lora(self, lora_id: int) -> bool:
    return self.model_executor.remove_lora(lora_id)

def list_loras(self) -> set[int]:
    return self.model_executor.list_loras()

def pin_lora(self, lora_id: int) -> bool:
    return self.model_executor.pin_lora(lora_id)
```

它不做 adapter 文件读取，不管理 LoRA slot，只把请求交给 model executor。

---

## 5. Executor 如何广播到 Worker

V1 executor 抽象类中统一实现 LoRA 控制面：`abstract.py:292` 到 `abstract.py:308`

```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
    return all(self.collective_rpc("add_lora", args=(lora_request,)))

def remove_lora(self, lora_id: int) -> bool:
    assert lora_id > 0, "lora_id must be greater than 0."
    return all(self.collective_rpc("remove_lora", args=(lora_id,)))

def pin_lora(self, lora_id: int) -> bool:
    assert lora_id > 0, "lora_id must be greater than 0."
    return all(self.collective_rpc("pin_lora", args=(lora_id,)))

def list_loras(self) -> set[int]:
    sets: list[set[int]] = self.collective_rpc("list_loras")
    for s in sets:
        assert s == sets[0], "All workers should have the same LORAs."
    return sets[0]
```

这里有几个重要语义：

```text
1. add/remove/pin 都要求 id > 0；
2. add/remove/pin 通过 collective_rpc 发到所有 worker；
3. 返回值使用 all(...) 聚合，任何 worker 返回 False，整体就是 False；
4. list_loras 要求所有 worker 返回完全相同的 LoRA id 集合，否则 assert。
```

这保证了一个 executor 内各 TP/PP worker 的 adapter 注册集合保持一致。

---

## 6. Worker 到 ModelRunner

`WorkerBase` 只定义接口：`worker_base.py:165` 到 `worker_base.py:175`

```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    raise NotImplementedError
...
```

GPU worker 实现是简单转发：`gpu_worker.py:958` 到 `gpu_worker.py:968`

```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    return self.model_runner.add_lora(lora_request)

def remove_lora(self, lora_id: int) -> bool:
    return self.model_runner.remove_lora(lora_id)

def list_loras(self) -> set[int]:
    return self.model_runner.list_loras()

def pin_lora(self, lora_id: int) -> bool:
    return self.model_runner.pin_lora(lora_id)
```

实际逻辑在 `LoRAModelRunnerMixin`：`lora_model_runner_mixin.py:274` 到 `lora_model_runner_mixin.py:288`

```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    self._ensure_lora_enabled()
    return self.lora_manager.add_adapter(lora_request)

def remove_lora(self, lora_id: int) -> bool:
    self._ensure_lora_enabled()
    return self.lora_manager.remove_adapter(lora_id)

def pin_lora(self, lora_id: int) -> bool:
    self._ensure_lora_enabled()
    return self.lora_manager.pin_adapter(lora_id)

def list_loras(self) -> set[int]:
    self._ensure_lora_enabled()
    return self.lora_manager.list_adapters()
```

如果模型没有启用 LoRA，会在 `_ensure_lora_enabled()` 抛错：

```python
if not hasattr(self, "lora_manager"):
    raise RuntimeError("LoRA is not enabled. Use --enable-lora to enable LoRA.")
```

位置：`lora_model_runner_mixin.py:69` 到 `lora_model_runner_mixin.py:71`

---

## 7. ModelRunner 初始化 LoRA manager

LoRA manager 在模型加载阶段创建。

`LoRAModelRunnerMixin.load_lora_model()`：`lora_model_runner_mixin.py:31` 到 `lora_model_runner_mixin.py:46`

```python
if not supports_lora(model):
    raise ValueError(f"{model.__class__.__name__} does not support LoRA yet.")

self.lora_manager = LRUCacheWorkerLoRAManager(
    vllm_config,
    device,
    model.embedding_modules,
)
return self.lora_manager.create_lora_manager(model, vllm_config)
```

GPUModelRunner 加载模型时调用：

```python
if self.lora_config:
    self.model = self.load_lora_model(
        self.model, self.vllm_config, self.device
    )
```

位置：`gpu_model_runner.py:5167` 到 `gpu_model_runner.py:5169`

因此：

```text
只有启动时启用了 LoRA 配置，ModelRunner 才会创建 lora_manager；
后续 add_lora/remove_lora/pin_lora/list_loras 都依赖这个 manager。
```

---

## 8. add_lora 的 Worker 侧生命周期

V1 默认使用 `LRUCacheWorkerLoRAManager`。

### 8.1 add_adapter()

`LRUCacheWorkerLoRAManager.add_adapter()` 定义在：`worker_manager.py:273`

```python
if (
    lora_request.lora_int_id not in self.list_adapters()
    or lora_request.load_inplace
):
    # Load the new adapter first to ensure it is actually valid, before
    # evicting any existing adapters.
    lora = self._load_adapter(lora_request)

    # Remove the existing adapter if it exists
    self._adapter_manager.remove_adapter(lora.id)

    # Loading succeeded, now check if we will exceed cache capacity and
    # evict if the oldest adapter if so
    if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
        self._adapter_manager.remove_oldest_adapter()

    loaded = self._adapter_manager.add_adapter(lora)
else:
    # If the lora is already loaded, just touch it to update its position
    loaded = (
        self._adapter_manager.get_adapter(lora_request.lora_int_id) is not None
    )
self._adapter_manager.activate_adapter(lora_request.lora_int_id)
return loaded
```

位置：`worker_manager.py:273` 到 `worker_manager.py:307`

这段逻辑很关键：

```text
1. 如果 adapter 不存在，或 load_inplace=True，先加载新 adapter；
2. 加载成功后，才删除已有同 ID adapter；
3. 如果超过 capacity，淘汰最老 adapter；
4. 把新 adapter 注册进 manager；
5. activate_adapter() 把它放入 GPU LoRA slot；
6. 如果 adapter 已存在，则不重新加载，只 touch LRU 并 activate。
```

### 8.2 _load_adapter()

`WorkerLoRAManager._load_adapter()` 定义在：`worker_manager.py:99`

核心流程：

```text
1. 根据 supported_lora_modules / packed_modules_mapping 计算 expected_lora_modules；
2. get_adapter_absolute_path(lora_request.lora_path) 解析路径；
3. PEFTHelper.from_local_dir() 读取 adapter config；
4. peft_helper.validate_legal(self.lora_config) 校验 rank、dtype、模块等合法性；
5. LoRAModel.from_local_checkpoint() 从 safetensors / bin / pt / tensorizer 读取权重；
6. 设置 lora.is_3d_lora_weight；
7. 返回 LoRAModel。
```

源码位置：`worker_manager.py:99` 到 `worker_manager.py:162`

如果路径缺失，会转成：

```python
raise LoRAAdapterNotFoundError(
    lora_request.lora_name, lora_request.lora_path
)
```

位置：`worker_manager.py:150` 到 `worker_manager.py:158`

### 8.3 LoRAModel.from_local_checkpoint()

`LoRAModel.from_local_checkpoint()` 定义在：`lora_model.py:166`

它会按优先级查找：

```text
adapter_model.safetensors
adapter_model.bin
adapter_model.pt
tensorizer adapter_model.tensors
```

相关位置：`lora_model.py:205` 到 `lora_model.py:295`

如果启用 MoE expert parallel，并传入 `moe_ep_spec`，会跳过非本 EP rank 的 expert 权重。

位置：`lora_model.py:268` 到 `lora_model.py:293`

---

## 9. adapter manager 的注册、激活和 slot

Worker manager 外层负责加载 checkpoint；底层 `LoRAModelManager` 负责注册和激活。

### 9.1 registered adapter cache

普通 `LoRAModelManager.add_adapter()`：

```python
if adapter.id in self._registered_adapters:
    return False
if len(self._registered_adapters) >= self.capacity:
    raise RuntimeError("No free adapter slots.")
self._add_adapter(adapter)
return True
```

位置：`model_manager.py:1130` 到 `model_manager.py:1137`

LRU 版本：

```python
if lora.id not in self._registered_adapters:
    self._add_adapter(lora)
    was_added = True
else:
    self._registered_adapters.touch(lora.id)
    was_added = False
return was_added
```

位置：`model_manager.py:1196` 到 `model_manager.py:1206`

其中：

```python
self._registered_adapters: LoRALRUCache = LoRALRUCache(
    self.capacity, self.deactivate_adapter
)
```

位置：`model_manager.py:1185` 到 `model_manager.py:1187`

这就是 CPU / manager 侧 adapter cache。

### 9.2 active GPU slot

`LoRAModelManager.activate_adapter()`：`model_manager.py:285`

```python
if lora_id in self._active_adapters:
    return False
first_free_slot = next(
    ((i, lora_id) for i, lora_id in enumerate(self.lora_index_to_id)
     if lora_id is None),
    None,
)
if first_free_slot is None:
    raise ValueError("No free lora slots")
index, _ = first_free_slot
self._active_adapters[lora_id] = None
lora_model = self._registered_adapters[lora_id]
self.lora_index_to_id[index] = lora_model.id
for module_name, module in self.modules.items():
    module_lora = self._get_lora_layer_weights(lora_model, module_name)
    if not module_lora:
        module.reset_lora(index)
        continue
    module.set_lora(index, module_lora.lora_a, module_lora.lora_b)
return True
```

位置：`model_manager.py:285` 到 `model_manager.py:324`

它做的是：

```text
registered adapter -> active GPU LoRA slot -> 各 LoRA wrapper 的 lora_a_stacked / lora_b_stacked。
```

LRU active cache 会在 slot 满时淘汰最老 active adapter：

```python
if (
    lora_id not in self._active_adapters
    and len(self._active_adapters) >= self.lora_slots
):
    self._active_adapters.remove_oldest()
result = super().activate_adapter(lora_id)
self._active_adapters.touch(lora_id)
return result
```

位置：`model_manager.py:1208` 到 `model_manager.py:1220`

因此有两个容量概念：

| 容量 | 字段 | 含义 |
|---|---|---|
| CPU / registered cache | `capacity = max_cpu_loras` | 最多注册多少个 adapter |
| GPU active slot | `lora_slots = max_loras` | 单次可激活多少个 LoRA slot |

---

## 10. remove_lora 生命周期

控制面链路：

```text
LLMEngine.remove_lora(lora_id)
  → EngineCore.remove_lora(lora_id)
  → Executor.remove_lora(lora_id)
  → collective_rpc("remove_lora")
  → GPUWorker.remove_lora(lora_id)
  → GPUModelRunner.remove_lora(lora_id)
  → lora_manager.remove_adapter(lora_id)
```

ModelRunner mixin：

```python
def remove_lora(self, lora_id: int) -> bool:
    self._ensure_lora_enabled()
    return self.lora_manager.remove_adapter(lora_id)
```

位置：`lora_model_runner_mixin.py:278` 到 `lora_model_runner_mixin.py:280`

底层 manager：

```python
def remove_adapter(self, adapter_id: int) -> bool:
    self.deactivate_adapter(adapter_id)
    if adapter_id not in self._registered_adapters:
        return False
    self._registered_adapters.pop(adapter_id, None)
    return True
```

位置：`model_manager.py:1144` 到 `model_manager.py:1149`

`deactivate_adapter()`：

```python
if adapter_id not in self._active_adapters:
    return False
self._deactivate_adapter(adapter_id)
self._active_adapters.pop(adapter_id, None)
return True
```

位置：`model_manager.py:1123` 到 `model_manager.py:1128`

`_deactivate_adapter()` 只清理 slot id：

```python
try:
    index = self.lora_index_to_id.index(lora_id)
    self.lora_index_to_id[index] = None
except ValueError:
    pass
```

位置：`model_manager.py:326` 到 `model_manager.py:331`

注意：

```text
remove_adapter() 没有检查当前是否还有 request_lora_mapping 指向这个 lora_id。
```

也就是说，remove 是控制面直接删除 adapter 的能力。调用方应避免在仍有正在执行或即将执行的请求依赖该 adapter 时删除它，否则后续请求如果还需要这个 LoRA，可能需要重新 add，或在 active adapter 查找时失败。

---

## 11. pin_lora 生命周期

控制面链路：

```text
LLMEngine.pin_lora(lora_id)
  → EngineCore.pin_lora(lora_id)
  → Executor.pin_lora(lora_id)
  → collective_rpc("pin_lora")
  → GPUWorker.pin_lora(lora_id)
  → GPUModelRunner.pin_lora(lora_id)
  → lora_manager.pin_adapter(lora_id)
```

ModelRunner mixin：

```python
def pin_lora(self, lora_id: int) -> bool:
    self._ensure_lora_enabled()
    return self.lora_manager.pin_adapter(lora_id)
```

位置：`lora_model_runner_mixin.py:282` 到 `lora_model_runner_mixin.py:284`

普通 `LoRAModelManager` 不支持 pin：

```python
raise NotImplementedError(
    "Pinning is not supported in LoRAModelManager. "
    "Use LRUCacheLoRAModelManager for pinning"
)
```

位置：`model_manager.py:337` 到 `model_manager.py:342`

LRU manager 支持：

```python
def pin_adapter(self, lora_id: int) -> bool:
    self._pin_lora_in_cpu_cache(lora_id)
    self._pin_lora_in_gpu_cache(lora_id)
    return True
```

位置：`model_manager.py:1228` 到 `model_manager.py:1232`

CPU cache pin：

```python
try:
    self._registered_adapters.pin(lora_id)
except ValueError as err:
    raise ValueError(
        f"Pinning failed. LoRA {lora_id} is not registered."
    ) from err
```

位置：`model_manager.py:1234` 到 `model_manager.py:1240`

GPU cache pin：

```python
if lora_id not in self._active_adapters:
    # move lora to gpu if not already active
    self.activate_adapter(lora_id)

self._active_adapters.pin(lora_id)
```

位置：`model_manager.py:1242` 到 `model_manager.py:1247`

因此：

```text
pin_lora 要求 adapter 已 registered；
pin 会确保 adapter 也 active 到 GPU slot；
pin 后它仍占用 max_cpu_loras 和 max_loras 资源，只是避免被 LRU 淘汰。
```

---

## 12. list_loras 生命周期

控制面链路：

```text
LLMEngine.list_loras()
  → EngineCore.list_loras()
  → Executor.list_loras()
  → collective_rpc("list_loras")
  → GPUWorker.list_loras()
  → GPUModelRunner.list_loras()
  → lora_manager.list_adapters()
```

ModelRunner mixin：

```python
def list_loras(self) -> set[int]:
    self._ensure_lora_enabled()
    return self.lora_manager.list_adapters()
```

位置：`lora_model_runner_mixin.py:286` 到 `lora_model_runner_mixin.py:288`

Worker manager：

```python
def list_adapters(self) -> set[int]:
    return set(self._adapter_manager.list_adapters())
```

位置：`worker_manager.py:227` 到 `worker_manager.py:228`

LRU manager：

```python
def list_adapters(self) -> dict[int, LoRAModel]:
    """List all registered LoRAModels."""
    return dict(self._registered_adapters.cache)
```

位置：`model_manager.py:1192` 到 `model_manager.py:1194`

Executor 会要求所有 worker 返回一致集合：

```python
for s in sets:
    assert s == sets[0], "All workers should have the same LORAs."
```

位置：`abstract.py:304` 到 `abstract.py:308`

所以 `list_loras()` 返回的是 registered adapter id 集合，不只是当前 active GPU slot 集合。

---

## 13. 请求数据面和控制面的关系

请求数据面在每轮执行前会调用：

```python
self.set_active_loras(
    self.input_batch, num_scheduled_tokens, num_sampled_tokens
)
```

位置：`gpu_model_runner.py:2193` 到 `gpu_model_runner.py:2201`

`set_active_loras()` 最终调用：

```python
self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
```

位置：`lora_model_runner_mixin.py:48` 到 `lora_model_runner_mixin.py:68`

`WorkerLoRAManager.set_active_adapters()`：

```python
def set_active_adapters(self, requests: set[Any], mapping: Any | None) -> None:
    self._apply_adapters(requests)
    if mapping is not None:
        self._adapter_manager.set_adapter_mapping(mapping)
```

位置：`worker_manager.py:183` 到 `worker_manager.py:186`

这意味着：

```text
即使用户没有显式调用 add_lora，只要请求携带 LoRARequest，
Worker 在 set_active_adapters() 时也会尝试按需加载 adapter。
```

对 LRU manager：

```python
for lora in loras_map.values():
    self.add_adapter(lora)
```

位置：`worker_manager.py:258` 到 `worker_manager.py:271`

因此控制面 `add_lora()` 主要用于预加载或显式管理；请求数据面也具备按需加载能力。

---

## 14. 失败和一致性语义

### 14.1 单 worker 加载失败

`_load_adapter()` 中任何加载或校验失败都会抛异常。

典型路径：

```python
except FileNotFoundError as e:
    raise LoRAAdapterNotFoundError(
        lora_request.lora_name, lora_request.lora_path
    ) from e
except Exception as e:
    raise e
```

位置：`worker_manager.py:150` 到 `worker_manager.py:160`

Executor 的 `add_lora()` 是：

```python
return all(self.collective_rpc("add_lora", args=(lora_request,)))
```

位置：`abstract.py:292` 到 `abstract.py:294`

如果某个 worker 抛异常，通常会由 collective RPC 向上抛出，而不是返回 False。源码里没有跨 worker 的显式事务回滚逻辑。

### 14.2 返回 False 的情况

可能返回 False 的常见情况：

```text
- add_lora：adapter 已存在且没有 load_inplace，底层 add_adapter 可能返回 False；
- remove_lora：adapter id 不存在；
- activate_adapter：adapter 已经 active 时内部可能返回 False，但外层 add_lora 仍可能返回 loaded 状态；
```

Executor 用 `all(...)` 聚合，所以只要一个 worker 返回 False，整体就是 False。

### 14.3 list_loras 一致性

`list_loras()` 明确 assert 所有 worker 集合一致：

```python
assert s == sets[0], "All workers should have the same LORAs."
```

位置：`abstract.py:305` 到 `abstract.py:308`

这不是修复机制，只是检测机制。如果 worker 状态已经不一致，会直接暴露错误。

---

## 15. add_lora 与正在执行 batch 的关系

源码注释说明 `LRUCacheWorkerLoRAManager.add_adapter()`：

```python
# Note that this method is not thread-safe. It may be invoked multiple
# times for the same adapter when using multiple API servers.
# This is ok because it's currently only called from
# the single-threaded core engine loop.
```

位置：`worker_manager.py:273` 到 `worker_manager.py:277`

这说明当前设计依赖 engine core loop 的串行控制来避免同一 worker 内的并发写 LoRA manager 状态。

同时，每轮模型执行前会重新：

```text
InputBatch.make_lora_inputs()
→ set_active_loras()
→ set_active_adapters()
→ set_adapter_mapping()
```

因此 add_lora 和请求执行的关系是：

```text
- add_lora 可以提前把 adapter 放进 registered / active cache；
- 如果请求已经携带 LoRARequest，即使没提前 add，也会在该 batch set_active_adapters 时按需加载；
- active mapping 是每轮 batch 重新设置的，adapter 是否已加载只是减少该轮加载开销。
```

---

## 16. remove_lora 与正在使用的 adapter

`remove_adapter()` 直接：

```text
deactivate_adapter(adapter_id)
→ 从 _registered_adapters 删除
```

位置：`model_manager.py:1144` 到 `model_manager.py:1149`

它没有读取：

```text
InputBatch.request_lora_mapping
lora_id_to_request_ids
Scheduler running / waiting queues
```

因此从源码看：

```text
remove_lora 是强控制接口，不会自动等待所有使用该 adapter 的请求完成。
```

如果用户在仍有请求使用该 LoRA 时删除它，后续行为取决于请求是否仍携带完整 `LoRARequest` 以及是否会再次触发按需加载。为了避免运行时抖动或失败，控制面调用者应在确认没有 in-flight 请求依赖该 adapter 后再 remove。

---

## 17. pin_lora 是否占用容量

是。

`pin_lora()` 会：

```text
1. pin registered adapter cache；
2. 如果 adapter 不在 active GPU cache，先 activate_adapter()；
3. pin active GPU cache。
```

位置：`model_manager.py:1228` 到 `model_manager.py:1247`

因此它会占用：

```text
max_cpu_loras 的一个 registered cache 槽；
max_loras 的一个 active GPU slot。
```

pin 的作用不是免费常驻，而是告诉 LRU cache：

```text
不要把这个 adapter 作为 oldest item 淘汰。
```

如果 pin 的 LoRA id 尚未 registered，会抛：

```python
ValueError(f"Pinning failed. LoRA {lora_id} is not registered.")
```

位置：`model_manager.py:1234` 到 `model_manager.py:1240`

---

## 18. shutdown / warmup / 清理

LoRA 还有两个生命周期相关点。

### 18.1 warmup dummy LoRA

`LoRAModelRunnerMixin` 有 dummy LoRA context，用于 warmup / capture：

```python
with self.lora_manager.dummy_lora_cache():
    for lr in lora_requests:
        self.lora_manager.add_dummy_lora(lr, rank=lora_warmup_rank)
    yield
...
if remove_lora:
    self.lora_manager.remove_all_adapters()
```

位置：`lora_model_runner_mixin.py:93` 到 `lora_model_runner_mixin.py:130`

`GPUWorker` warmup 后也会清理：

```python
self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)
```

位置：`gpu_worker.py:618` 到 `gpu_worker.py:622`

### 18.2 remove_all_adapters()

`LoRAModelManager.remove_all_adapters()`：

```python
self._registered_adapters.clear()
self.lora_index_to_id = [None] * self.lora_slots
self._active_adapters.clear()
```

位置：`model_manager.py:369` 到 `model_manager.py:373`

`LoRAModelRunnerMixin.maybe_remove_all_loras()`：

```python
def maybe_remove_all_loras(self, lora_config: LoRAConfig | None):
    if lora_config is None:
        return
    self.lora_manager.remove_all_adapters()
```

位置：`lora_model_runner_mixin.py:269` 到 `lora_model_runner_mixin.py:272`

Executor shutdown 本身是通用 worker shutdown：

```python
def shutdown(self) -> None:
    self.collective_rpc("shutdown")
```

位置：`abstract.py:276` 到 `abstract.py:278`

没有单独的 LoRA shutdown 协议；adapter 生命周期随 worker / model runner 生命周期结束而释放。

---

## 19. 一轮请求触发按需加载的例子

如果没有提前调用 `add_lora()`，但请求带了：

```python
LoRARequest(lora_name="sql", lora_int_id=1, lora_path="/path/to/sql-lora")
```

本轮执行前会走：

```text
SchedulerOutput.scheduled_new_reqs
  → CachedRequestState.lora_request
  → InputBatch.add_request()
  → lora_id_to_lora_request[1] = LoRARequest(...)
  → InputBatch.make_lora_inputs()
  → active_lora_requests = {LoRARequest(...)}
  → set_active_loras()
  → LRUCacheWorkerLoRAManager._apply_adapters()
  → add_adapter(LoRARequest(...))
  → _load_adapter("/path/to/sql-lora")
  → activate_adapter(1)
  → set_adapter_mapping(token_lora_mapping)
```

这说明：

```text
add_lora 是预加载控制接口；
请求数据面也会触发 adapter 按需加载。
```

---

## 20. 总结

LoRA 生命周期可以按状态分成四层：

```text
1. 请求绑定：
   LoRARequest 挂在请求上，表示这个请求要使用哪个 adapter。

2. registered cache：
   LoRAModel 已从 checkpoint 加载并注册到 LoRAModelManager。
   受 max_cpu_loras / LRU / pin 影响。

3. active GPU slot：
   LoRA 权重已写入各 LoRA wrapper 的 lora_a_stacked / lora_b_stacked slot。
   受 max_loras / LRU / pin 影响。

4. per-batch mapping：
   本轮 token_lora_mapping / prompt_lora_mapping 指向具体 lora_int_id，
   Punica wrapper 根据 mapping 在 forward 中应用 LoRA delta。
```

控制接口对应的作用是：

| 接口 | 作用 |
|---|---|
| `add_lora` | 预加载并激活 adapter，必要时触发 LRU 淘汰 |
| `remove_lora` | 取消 active slot 并从 registered cache 删除 |
| `pin_lora` | 固定 registered cache 和 active GPU cache，避免 LRU 淘汰 |
| `list_loras` | 查询所有 worker 一致的 registered adapter id 集合 |

最重要的边界是：

```text
LoRA 控制面管理 adapter 缓存和 slot；
Scheduler 请求数据面管理每个 request 用哪个 lora_int_id；
真正 forward 前，ModelRunner 把两者汇合成 LoRAMapping。
```
