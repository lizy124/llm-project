# 04. Worker / ModelRunner 如何维护 active LoRA 状态？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/worker_base.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py`
- `code/vllm/vllm/lora/request.py`
- `code/vllm/vllm/lora/worker_manager.py`
- `code/vllm/vllm/lora/model_manager.py`
- `code/vllm/vllm/lora/layers/utils.py`
- `code/vllm/vllm/lora/layers/`
- `code/vllm/vllm/lora/punica_wrapper/`

本文关注：LoRA adapter 已经能被 Worker 侧 manager 加载后，每轮执行前，`Worker / GPUModelRunner / InputBatch` 如何知道当前 batch 中每个 request 使用哪个 LoRA，如何把 request 级 `LoRARequest` 翻译成 batch/token 级 `LoRAMapping`，并最终把 active mapping 同步给 LoRA layer / punica wrapper。

它不展开 LoRA checkpoint 如何加载、layer 如何注入、kernel 如何计算 delta；这些分别放在 `03_lora_manager_and_cache.md`、`05_lora_layer_injection.md`、`06_batch_mixed_lora_execution.md`。

---

## 0. 梳理规划

本文按“先定状态边界，再走每轮执行链路，再拆关键数据结构，最后总结特殊场景”的顺序组织。

要回答的问题分成 9 组：

```text
1. active LoRA 状态处在 Worker / ModelRunner 的哪一层？
2. LoRARequest 如何进入 CachedRequestState？
3. InputBatch 保存哪些 LoRA 状态？
4. request_lora_mapping / lora_id_to_request_ids / lora_id_to_lora_request 分别是什么？
5. _prepare_inputs() 什么时候调用 set_active_loras()？
6. make_lora_inputs() 如何生成 prompt_lora_mapping 和 token_lora_mapping？
7. LoRAModelRunnerMixin 如何把 mapping 交给 lora_manager？
8. WorkerLoRAManager / LoRAModelManager 如何更新 active adapters 和 punica metadata？
9. batch reorder、request remove、spec decode、multimodal tower LoRA 会怎样影响 LoRA 状态？
```

阅读顺序建议：

```text
lora_and_adapters_overview.md
  → 02_lora_request_and_engine_flow.md
  → 03_lora_manager_and_cache.md
  → 04_worker_model_runner_lora_state.md
  → 06_batch_mixed_lora_execution.md
  → 05_lora_layer_injection.md
```

如果只想抓主线，可以先看第 2、3、4、5、6 节。

---

## 1. 一句话回答

```text
Worker / ModelRunner 维护 active LoRA 状态的核心，是把“每个 request 使用哪个 LoRARequest”
转换成“本轮 flat token batch 中每个 token / sampled token 使用哪个 LoRA id”。

具体做法是：
GPUModelRunner 在 _update_states() 中把 LoRARequest 存入 CachedRequestState；
InputBatch.add_request() 把 request 级 LoRA id 记录到 request_lora_mapping；
_prepare_inputs() 已经确定本轮每个 request 的 scheduled token 数后，调用 set_active_loras()；
InputBatch.make_lora_inputs() 生成 prompt_lora_mapping、token_lora_mapping 和 active LoRARequest 集合；
LoRAModelRunnerMixin 把这些包装成 LoRAMapping；
WorkerLoRAManager 确保 adapter 已加载并把 mapping 写入 LoRAModelManager / punica wrapper。
```

压缩成一条链路是：

```text
NewRequestData.lora_request
  → CachedRequestState.lora_request
  → InputBatch.request_lora_mapping[req_index]
  → InputBatch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)
  → LoRAMapping(index_mapping, prompt_mapping)
  → lora_manager.set_active_adapters(lora_requests, mapping)
  → LoRAModelManager.set_adapter_mapping(mapping)
  → punica_wrapper.update_metadata(...)
  → LoRA layer forward 使用当前 mapping
```

---

## 2. active LoRA 状态的边界

LoRA 在 Worker / ModelRunner 侧至少有三类状态，不能混在一起。

### 2.1 adapter 资源状态

这是 manager 管的状态：

```text
- 哪些 adapter 已注册；
- 哪些 adapter 已加载到 GPU slot；
- 哪些 adapter 被 pin；
- LoRA 权重在哪些 LoRA-wrapped layer 的 slot 上。
```

主要在：

- `code/vllm/vllm/lora/worker_manager.py`
- `code/vllm/vllm/lora/model_manager.py`

这部分是 `03_lora_manager_and_cache.md` 的重点。

### 2.2 request 级 LoRA 状态

这是每个请求自己的状态：

```text
request.lora_request
CachedRequestState.lora_request
```

它回答：

```text
这个 request 想用哪个 adapter？
```

### 2.3 batch / token 级 active mapping

这是本文重点。

它回答：

```text
本轮 forward 的 flat token batch 中，每个 token 应该使用哪个 LoRA id？
本轮 sampler / logits 相关的 sampled token 位置，又应该使用哪个 LoRA id？
```

相关对象是：

```text
InputBatch.request_lora_mapping
InputBatch.lora_id_to_request_ids
InputBatch.lora_id_to_lora_request
LoRAMapping.index_mapping
LoRAMapping.prompt_mapping
```

一句话区分：

```text
LoRA manager 负责“有哪些 adapter 权重可用”；
InputBatch / ModelRunner 负责“这一轮哪些 token 使用哪些 adapter”。
```

---

## 3. ModelRunner 初始化 LoRA 能力

`GPUModelRunner` 继承了 `LoRAModelRunnerMixin`。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:418`

```text
class GPUModelRunner(LoRAModelRunnerMixin, ...):
```

初始化时保存配置：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:431`

```text
self.lora_config = vllm_config.lora_config
```

真正创建 LoRA manager 的位置在模型加载阶段。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5167`

```text
if self.lora_config:
    self.model = self.load_lora_model(
        self.model, self.vllm_config, self.device
    )
```

`load_lora_model()` 来自 `LoRAModelRunnerMixin`。

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:31`

它会：

```text
1. 检查模型是否 supports_lora；
2. 创建 LRUCacheWorkerLoRAManager；
3. 调用 manager.create_lora_manager(model, vllm_config)；
4. 返回被 LoRA manager 包装后的 model。
```

核心代码是：

```text
self.lora_manager = LRUCacheWorkerLoRAManager(...)
return self.lora_manager.create_lora_manager(model, vllm_config)
```

这一步完成后，ModelRunner 才具备：

```text
add_lora / remove_lora / pin_lora / list_loras
set_active_loras
maybe_dummy_run_with_lora
```

如果没有启用 LoRA，调用这些接口会触发：

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:69`

```text
RuntimeError("LoRA is not enabled. Use --enable-lora to enable LoRA.")
```

---

## 4. LoRARequest 如何进入 worker 侧 request state

`LoRARequest` 的核心字段是：

源码位置：`code/vllm/vllm/lora/request.py:8`

```text
lora_name: str
lora_int_id: int
lora_path: str
```

其中：

```text
lora_int_id 必须 > 0；
adapter_id 属性返回 lora_int_id。
```

Worker 侧首次接收新请求时，`SchedulerOutput.scheduled_new_reqs` 中的 `NewRequestData` 会携带 `lora_request`。

`GPUModelRunner._update_states()` 会把它写入 `CachedRequestState`。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1194`

核心关系是：

```text
NewRequestData.lora_request
  → CachedRequestState(lora_request=new_req_data.lora_request)
  → self.requests[req_id] = req_state
```

对应代码字段：

```text
req_state = CachedRequestState(
    req_id=req_id,
    ...
    lora_request=new_req_data.lora_request,
)
```

`CachedRequestState` 的定义在：

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:33`

其中有：

```text
lora_request: LoRARequest | None = None
```

这说明：

```text
LoRARequest 首先是 request 级状态，而不是 token 级状态。
```

---

## 5. InputBatch 维护哪些 LoRA 状态

`InputBatch` 是 worker 侧 persistent batch 状态。

LoRA 相关字段在初始化时创建。

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:244`

```text
self.request_lora_mapping = np.zeros((self.max_num_reqs,), dtype=np.int64)
self.lora_id_to_request_ids: dict[int, set[str]] = {}
self.lora_id_to_lora_request: dict[int, LoRARequest] = {}
```

它们各自的含义是：

### 5.1 `request_lora_mapping`

```text
request_lora_mapping[req_index] = lora_id
```

它是 request index 到 LoRA id 的映射。

约定是：

```text
0：无 LoRA；
>0：LoRARequest.lora_int_id。
```

因为 `LoRARequest.__post_init__()` 要求 `lora_int_id >= 1`，所以 `0` 可以安全地表示 no-LoRA。

### 5.2 `lora_id_to_request_ids`

```text
lora_id_to_request_ids[lora_id] = {req_id, ...}
```

它记录当前 batch 中有哪些 request 正在使用某个 LoRA id。

用途是：

```text
当 request 被移除时，判断这个 LoRA id 是否还有其它 request 在用。
```

### 5.3 `lora_id_to_lora_request`

```text
lora_id_to_lora_request[lora_id] = LoRARequest
```

它保存当前 batch 中 active LoRA id 对应的完整 `LoRARequest`。

用途是：

```text
set_active_loras() 时交给 manager，manager 可以据此加载或保留对应 adapter。
```

一句话记忆：

```text
request_lora_mapping 管“第几个 request 用哪个 id”；
lora_id_to_request_ids 管“这个 id 还有谁在用”；
lora_id_to_lora_request 管“这个 id 对应哪个 LoRARequest”。
```

---

## 6. add_request() 如何注册 LoRA

当 `_update_states()` 把请求加入 persistent batch 时，会调用 `InputBatch.add_request()`。

LoRA 注册逻辑在函数尾部。

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:468`

如果 request 有 LoRA：

```text
lora_id = request.lora_request.lora_int_id
if lora_id not in self.lora_id_to_request_ids:
    self.lora_id_to_request_ids[lora_id] = set()

self.request_lora_mapping[req_index] = lora_id
self.lora_id_to_request_ids[lora_id].add(request.req_id)
self.lora_id_to_lora_request[lora_id] = request.lora_request
```

如果没有 LoRA：

```text
self.request_lora_mapping[req_index] = 0
```

这一步完成后，InputBatch 只知道：

```text
当前 persistent batch 中每个 req_index 对应哪个 lora_id。
```

它还没有生成本轮 token 级 mapping，因为本轮每个 request schedule 多少 token 要等 `_prepare_inputs()` 才确定。

---

## 7. remove / swap / condense 如何维护 LoRA 状态

InputBatch 是 persistent batch，请求会被移除、交换、压缩。

LoRA 状态必须跟着 request row 一起移动。

### 7.1 `remove_request()`

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:510`

移除请求时，会读取该 row 的 LoRA id：

```text
lora_id = self.request_lora_mapping[req_index]
```

如果不是 0：

```text
lora_req_ids = self.lora_id_to_request_ids[lora_id]
lora_req_ids.discard(req_id)
if not lora_req_ids:
    del self.lora_id_to_request_ids[lora_id]
    del self.lora_id_to_lora_request[lora_id]
self.request_lora_mapping[req_index] = 0
```

含义是：

```text
只要 batch 中还有其它 request 使用同一个 LoRA id，
这个 LoRARequest 就仍然保留在 lora_id_to_lora_request 中；
最后一个 request 离开后，才从 batch active LoRA 集合中删除。
```

### 7.2 `swap_states()`

请求 row 交换时，LoRA mapping 也要交换。

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:630`

```text
self.request_lora_mapping[i1], self.request_lora_mapping[i2] = (
    self.request_lora_mapping[i2],
    self.request_lora_mapping[i1],
)
```

### 7.3 `condense()`

batch 压缩时，会把尾部非空 row 移到前面的空洞。

LoRA id 也跟着 row 移动：

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:759`

```text
self.request_lora_mapping[empty_index] = self.request_lora_mapping[last_req_index]
```

这些维护保证了一个不变量：

```text
request_lora_mapping 的下标永远和当前 InputBatch row / req_index 对齐。
```

---

## 8. `_prepare_inputs()` 什么时候激活 LoRA

LoRA active mapping 不是在 `_update_states()` 里设置，而是在 `_prepare_inputs()` 后段设置。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2193`

```text
if self.lora_config:
    assert (
        np.sum(num_sampled_tokens)
        <= self.vllm_config.scheduler_config.max_num_batched_tokens
    )
    self.set_active_loras(
        self.input_batch, num_scheduled_tokens, num_sampled_tokens
    )
```

为什么在这里？

因为此时已经确定了本轮执行所需的关键数组：

```text
num_scheduled_tokens：每个 request 本轮 forward 的 token 数；
num_sampled_tokens：每个 request 本轮会采样 / 接收的 token 数；
InputBatch.req_ids / req_id_to_index：本轮 batch 顺序；
request_lora_mapping：每个 req_index 对应的 LoRA id。
```

active LoRA mapping 依赖这些信息，所以必须等 `_prepare_inputs()` 已经把本轮 batch 展开之后才能生成。

---

## 9. make_lora_inputs() 如何生成 token mapping

`set_active_loras()` 会调用 `InputBatch.make_lora_inputs()`。

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:976`

签名是：

```text
make_lora_inputs(
    num_scheduled_tokens: np.ndarray,
    num_sampled_tokens: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]
```

它做的事情非常直接：

```text
req_lora_mapping = self.request_lora_mapping[: self.num_reqs]
prompt_lora_mapping = tuple(req_lora_mapping.repeat(num_sampled_tokens))
token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))
active_lora_requests = set(self.lora_id_to_lora_request.values())
```

### 9.1 `token_lora_mapping`

```text
token_lora_mapping: tuple[int, ...]
```

长度是：

```text
sum(num_scheduled_tokens)
```

它描述本轮 forward 的 flat input token batch 中，每个 token 用哪个 LoRA id。

例如：

```text
req0: no LoRA, scheduled 2 tokens
req1: LoRA 7, scheduled 3 tokens
req2: LoRA 9, scheduled 1 token

request_lora_mapping = [0, 7, 9]
num_scheduled_tokens = [2, 3, 1]

token_lora_mapping = [0, 0, 7, 7, 7, 9]
```

这和 `_prepare_inputs()` 生成的 flat token 顺序一致。

### 9.2 `prompt_lora_mapping`

```text
prompt_lora_mapping: tuple[int, ...]
```

长度是：

```text
sum(num_sampled_tokens)
```

它描述本轮 sampled token / logits 相关位置使用哪个 LoRA id。

普通非 spec decode 场景下：

```text
num_sampled_tokens = [1, 1, 1, ...]
```

所以 `prompt_lora_mapping` 通常是一 request 一个 id。

spec decode 场景下，`num_sampled_tokens` 可能是：

```text
num_draft_tokens + 1
```

因此同一个 request 可能在 sampled-token 维度重复多个 LoRA id。

### 9.3 `active_lora_requests`

```text
active_lora_requests = set(self.lora_id_to_lora_request.values())
```

它是当前 batch 中所有非 0 LoRA id 对应的 `LoRARequest` 集合。

这个集合交给 manager 后，manager 会确保这些 adapter 已加载，并移除不再 active 的 adapter。

---

## 10. LoRAModelRunnerMixin 如何设置 active LoRA

`GPUModelRunner.set_active_loras()` 来自 `LoRAModelRunnerMixin`。

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:73`

主逻辑是：

```text
prompt_lora_mapping, token_lora_mapping, lora_requests = (
    input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)
)
return self._set_active_loras(
    prompt_lora_mapping,
    token_lora_mapping,
    lora_requests,
    mapping_type,
)
```

`_set_active_loras()` 会创建 `LoRAMapping`。

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:48`

```text
lora_mapping = LoRAMapping(
    token_lora_mapping,
    prompt_lora_mapping,
    is_prefill=True,
    type=mapping_type,
)
self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
```

`LoRAMapping` 定义在：

源码位置：`code/vllm/vllm/lora/layers/utils.py:27`

```text
class LoRAMappingType(Enum):
    LANGUAGE = 1
    TOWER = 2
    CONNECTOR = 3

@dataclass
class LoRAMapping:
    index_mapping: tuple[int, ...]
    prompt_mapping: tuple[int, ...]
    is_prefill: bool = False
    type: LoRAMappingType = LoRAMappingType.LANGUAGE
```

在语言模型主 forward 中，默认 mapping type 是：

```text
LoRAMappingType.LANGUAGE
```

---

## 11. manager 如何应用 active adapters 和 mapping

`lora_manager.set_active_adapters()` 在 `WorkerLoRAManager` 中实现。

源码位置：`code/vllm/vllm/lora/worker_manager.py:183`

```text
self._apply_adapters(requests)
if mapping is not None:
    self._adapter_manager.set_adapter_mapping(mapping)
```

这里分两步。

### 11.1 `_apply_adapters()` 确保 adapter 资源就绪

源码位置：`code/vllm/vllm/lora/worker_manager.py:194`

它会：

```text
1. 取当前已经加载的 adapter ids；
2. 从本轮 lora_requests 构造 requested_ids；
3. 如果 requested_ids 数量超过 GPU adapter slots，报错；
4. 对已经存在但本轮不需要的 adapter，remove_adapter；
5. 对本轮需要但还没加载的 adapter，add_adapter。
```

简化逻辑是：

```text
existing_adapters = self.list_adapters()
requested_ids = set(models_map)

for adapter_id in existing_adapters - requested_ids:
    self.remove_adapter(adapter_id)
for adapter_id in requested_ids - existing_adapters:
    self.add_adapter(models_map[adapter_id])
```

对于 `LRUCacheWorkerLoRAManager`，底层 model manager 支持 LRU cache；但从 active state 视角看，关键是不变量：

```text
本轮 mapping 中出现的非 0 LoRA id，必须对应已加载 / 已激活 adapter。
```

### 11.2 `set_adapter_mapping()` 更新 punica metadata

`WorkerLoRAManager` 内部持有 `LoRAModelManager`。

`LoRAModelManager.set_adapter_mapping()` 在 mapping 变化时才更新。

源码位置：`code/vllm/vllm/lora/model_manager.py:1139`

```text
if self._last_mapping != mapping:
    self._set_adapter_mapping(mapping)
    self._last_mapping = mapping
```

真正更新 metadata 的位置：

源码位置：`code/vllm/vllm/lora/model_manager.py:344`

```text
punica_wrapper.update_metadata(
    mapping,
    self.lora_index_to_id,
    self.lora_slots + 1,
    self.vocab_size,
)
```

这里传入的几个东西含义是：

```text
mapping：本轮 token / prompt 到 LoRA id 的映射；
lora_index_to_id：GPU LoRA slot index 到 LoRA id 的映射；
lora_slots + 1：包含 no-LoRA / padding 语义的 slot 数；
vocab_size：embedding / logits LoRA 可能需要处理额外 vocab。
```

`punica_wrapper` 后续会被 LoRA-wrapped layer 使用。

---

## 12. LoRA layer 如何看到 active mapping

模型加载 LoRA 时，`LoRAModelManager` 会把目标模块替换成 LoRA wrapper。

在创建 wrapper 时，会调用：

源码位置：`code/vllm/vllm/lora/model_manager.py:499`

```text
new_module.set_mapping(punica_wrapper)
```

这意味着：

```text
每个 LoRA layer 不直接持有每轮 mapping 的 Python list；
它持有一个 punica wrapper 引用。
```

每轮 `_prepare_inputs()` 更新的是 punica wrapper 的 metadata。

forward 时 LoRA layer 通过这个 wrapper 读取当前 batch mapping，计算对应 LoRA delta。

这就是 active LoRA 状态最终生效的位置：

```text
InputBatch / ModelRunner 生成 mapping；
LoRAModelManager 写入 punica wrapper；
LoRA layer forward 使用 punica wrapper metadata。
```

---

## 13. 一个 batch 示例

假设当前 batch 有 4 个请求：

```text
req0: no LoRA
req1: LoRA A, lora_int_id = 3
req2: LoRA A, lora_int_id = 3
req3: LoRA B, lora_int_id = 8
```

`InputBatch.add_request()` 后：

```text
request_lora_mapping = [0, 3, 3, 8]

lora_id_to_request_ids = {
  3: {req1, req2},
  8: {req3},
}

lora_id_to_lora_request = {
  3: LoRARequest(A),
  8: LoRARequest(B),
}
```

如果本轮 schedule token 数是：

```text
num_scheduled_tokens = [1, 4, 2, 1]
num_sampled_tokens = [1, 1, 1, 1]
```

那么：

```text
token_lora_mapping = [
  0,
  3, 3, 3, 3,
  3, 3,
  8,
]

prompt_lora_mapping = [0, 3, 3, 8]

active_lora_requests = {LoRARequest(A), LoRARequest(B)}
```

这表示：

```text
- req0 的 token 只走 base model；
- req1 / req2 的 token 使用 LoRA A；
- req3 的 token 使用 LoRA B；
- 同一个 forward 不需要拆成三个模型 forward。
```

---

## 14. 和 speculative decoding 的关系

LoRA token mapping 和 spec decode 的交点在 `num_sampled_tokens`。

在 `_prepare_inputs()` 中：

```text
if not use_spec_decode:
    num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
else:
    num_sampled_tokens = num_draft_tokens + 1
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2153`

所以：

```text
token_lora_mapping：跟 num_scheduled_tokens 走，描述 forward 输入 tokens；
prompt_lora_mapping：跟 num_sampled_tokens 走，描述采样 / draft acceptance 相关位置。
```

spec decode 下，一个 request 可能对应多个 sampled positions，因此它的 LoRA id 会在 `prompt_lora_mapping` 中重复多次。

---

## 15. 和 CUDA graph dispatch 的关系

LoRA active state 还会影响 CUDA graph 的 batch dispatch。

`_determine_batch_execution_and_padding()` 会读取：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3845`

```text
num_active_loras = len(self.input_batch.lora_id_to_lora_request)
has_lora = num_active_loras > 0
```

然后把 `has_lora` 和 `num_active_loras` 交给 cudagraph dispatcher：

```text
self.cudagraph_dispatcher.dispatch(
    num_tokens=num_tokens,
    has_lora=has_lora,
    num_active_loras=num_active_loras,
    ...
)
```

这说明：

```text
LoRA 不只是 forward 内部 delta 计算的状态，
它也会影响这一轮选择哪个 CUDA graph / batch descriptor。
```

---

## 16. 和 multimodal tower / connector LoRA 的关系

本文主线讲的是 language model 的 `LoRAMappingType.LANGUAGE`。

多模态 encoder 还有额外分支。

`_execute_mm_encoder()` 中，如果模型和 manager 支持 tower / connector LoRA，会单独为 encoder batch 构造 mapping。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2941`

核心原因是：

```text
encoder batch 的结构按多模态 item 组织，
不是按 language model 的 scheduled token batch 组织，
所以不能复用 InputBatch.make_lora_inputs() 生成的 language token mapping。
```

它会单独构造：

```text
LoRAMappingType.TOWER
LoRAMappingType.CONNECTOR
```

并调用：

```text
self.lora_manager.set_active_adapters(lora_requests, tower_mapping)
self.lora_manager.set_active_adapters(lora_requests, connector_mapping)
```

这里仍然复用一个基础事实：

```text
每个 request 的 LoRA id 仍来自 InputBatch.request_lora_mapping[req_idx]。
```

只是 token 数量和 batch 维度由 multimodal encoder 的 item / embedding 数决定。

---

## 17. Worker 控制接口和 active state 的关系

Worker 暴露 LoRA 控制接口。

抽象接口在：

源码位置：`code/vllm/vllm/v1/worker/worker_base.py:165`

```text
add_lora
remove_lora
pin_lora
list_loras
```

GPU Worker 实现只是转发给 ModelRunner。

源码位置：`code/vllm/vllm/v1/worker/gpu_worker.py:958`

```text
def add_lora(self, lora_request: LoRARequest) -> bool:
    return self.model_runner.add_lora(lora_request)

def remove_lora(self, lora_id: int) -> bool:
    return self.model_runner.remove_lora(lora_id)

def list_loras(self) -> set[int]:
    return self.model_runner.list_loras()

def pin_lora(self, lora_id: int) -> bool:
    return self.model_runner.pin_lora(lora_id)
```

这些接口管理的是 adapter 生命周期。

而每轮 active state 仍由：

```text
InputBatch.make_lora_inputs()
  → set_active_loras()
```

决定。

两者关系是：

```text
控制接口让 adapter 可以存在于 manager/cache 中；
active mapping 决定本轮 forward 实际使用哪些 adapter。
```

---

## 18. warmup / dummy run 中的 LoRA 状态

LoRA 还会参与 warmup 和 CUDA graph capture。

`LoRAModelRunnerMixin` 提供：

```text
maybe_setup_dummy_loras()
maybe_select_dummy_loras()
maybe_dummy_run_with_lora()
maybe_remove_all_loras()
```

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:93`

这些函数会创建 dummy LoRARequest，并构造 dummy mapping。

作用是：

```text
在 warmup / graph capture 阶段模拟不同数量 active LoRA 的 batch，
让后续真实请求可以复用已经准备好的 kernel / graph 路径。
```

GPU Worker warmup 后会清理 dummy LoRA：

源码位置：`code/vllm/vllm/v1/worker/gpu_worker.py:621`

```text
self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)
```

---

## 19. 关键不变量

梳理 Worker / ModelRunner LoRA 状态时，可以抓住这些不变量。

### 19.1 `0` 表示 no-LoRA

```text
LoRARequest.lora_int_id >= 1；
InputBatch.request_lora_mapping 中的 0 表示不使用 LoRA。
```

### 19.2 `request_lora_mapping` 始终按 req_index 对齐

无论 add、remove、swap、condense，`request_lora_mapping[i]` 都必须对应当前 row `i` 的 request。

### 19.3 token mapping 只能在本轮 token 数确定后生成

```text
request_lora_mapping 是 request 级；
token_lora_mapping 是本轮 token 级。
```

`token_lora_mapping` 依赖 `num_scheduled_tokens`，所以在 `_prepare_inputs()` 后段生成。

### 19.4 active_lora_requests 是 batch 当前非 0 LoRA 集合

```text
set(self.lora_id_to_lora_request.values())
```

这个集合决定 manager 需要加载 / 保留哪些 adapter。

### 19.5 mapping 变化才更新 punica metadata

`LoRAModelManager.set_adapter_mapping()` 会缓存 `_last_mapping`，避免重复更新相同 mapping。

### 19.6 active adapter 和 active mapping 是两件事

```text
active adapter：权重是否在 manager / GPU slot 中可用；
active mapping：本轮每个 token 使用哪个 adapter。
```

两者都正确，LoRA forward 才正确。

---

## 20. 常见问题

### 20.1 为什么不能只设置一个全局 active LoRA？

因为同一个 batch 可以混合：

```text
- no-LoRA 请求；
- LoRA A 请求；
- LoRA B 请求；
- 同一个 LoRA 的多个请求；
- prefill / decode / spec decode 不同 token 数。
```

所以必须有 token 级 mapping。

### 20.2 为什么 `InputBatch` 要保存 `lora_id_to_request_ids`？

因为多个 request 可能共用同一个 LoRA id。

移除一个 request 时，不能立刻删除 `lora_id_to_lora_request`；只有最后一个使用该 LoRA id 的 request 离开 batch，才删除对应 active request 记录。

### 20.3 为什么 manager 会移除本轮不需要的 adapter？

`WorkerLoRAManager._apply_adapters()` 会比较 existing adapters 和 requested adapters。

这样可以让 manager 的 active slots 和当前 batch 需求对齐，并避免 GPU LoRA slot 被无关 adapter 占用。

对于 LRU manager，底层还会结合 cache 容量处理注册 / 激活 / 淘汰。

### 20.4 `prompt_lora_mapping` 为什么不是 prompt token 的 mapping？

这里名字容易误导。

在当前 V1 代码中，`prompt_lora_mapping` 的长度是 `sum(num_sampled_tokens)`，用于 sampled token / logits processor 相关位置。

`token_lora_mapping` 才是本轮 forward 输入 token 的 mapping，长度是 `sum(num_scheduled_tokens)`。

### 20.5 LoRA id 和 GPU slot index 是一回事吗？

不是。

```text
LoRA id：LoRARequest.lora_int_id，对外稳定标识 adapter；
GPU slot index：LoRAModelManager.lora_index_to_id 中的位置，表示这个 adapter 被放在哪个 LoRA 权重 slot。
```

`punica_wrapper.update_metadata()` 会同时拿到 mapping 和 `lora_index_to_id`，从而把 LoRA id 转成实际 slot。

---

## 21. 一句话总结

Worker / ModelRunner 的 active LoRA 状态维护，本质是一个两阶段翻译过程：

```text
阶段一：请求进入 batch
  LoRARequest
    → CachedRequestState.lora_request
    → InputBatch.request_lora_mapping[req_index]

阶段二：本轮执行前
  request_lora_mapping + num_scheduled_tokens + num_sampled_tokens
    → token_lora_mapping / prompt_lora_mapping / active_lora_requests
    → LoRAMapping
    → lora_manager.set_active_adapters()
    → punica_wrapper metadata
    → LoRA layer forward
```

最核心的主线是：

```text
GPUModelRunner._update_states()
  → InputBatch.add_request()
  → request_lora_mapping / lora_id_to_lora_request
  → GPUModelRunner._prepare_inputs()
  → LoRAModelRunnerMixin.set_active_loras()
  → InputBatch.make_lora_inputs()
  → LoRAMapping
  → WorkerLoRAManager.set_active_adapters()
  → LoRAModelManager.set_adapter_mapping()
  → punica_wrapper.update_metadata()
  → LoRA layer 使用当前 batch mapping
```
