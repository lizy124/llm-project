# 06. 同一 batch 中多个 LoRA 如何混合执行？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py`
- `code/vllm/vllm/lora/worker_manager.py`
- `code/vllm/vllm/lora/model_manager.py`
- `code/vllm/vllm/lora/layers/`
- `code/vllm/vllm/lora/punica_wrapper/`
- `code/vllm/vllm/lora/ops/triton_ops/`

本问题关注：同一个 batch 中不同请求使用不同 LoRA adapter 时，vLLM 如何避免拆成多个模型 forward，而是在一次 base model forward 中按 token / request 映射叠加不同 LoRA delta。

本文按“先定边界，再走主链路，再拆数据结构和 kernel 执行”的方式梳理。

---

## 0. 梳理规划

batch mixed LoRA 要回答的核心问题是：

```text
同一轮调度中：
  req0: no LoRA
  req1: LoRA A
  req2: LoRA A
  req3: LoRA B
  req4: no LoRA

vLLM 为什么不需要拆成：
  base-only forward + LoRA A forward + LoRA B forward？
```

答案是：

```text
vLLM 把“请求级 adapter 选择”转成“本轮 token 级 LoRA slot 映射”，
LoRA layer 在同一次 forward 中用 Punica metadata 为每个 token 选择对应 adapter 权重。
```

要回答的问题分成 8 组：

```text
1. request 级 LoRARequest 如何进入 InputBatch？
2. InputBatch 如何保存每个 request 的 LoRA id？
3. 每轮 _prepare_inputs() 如何生成 token_lora_mapping / prompt_lora_mapping？
4. LoRA manager 如何保证 adapter 已加载并映射到 GPU slot？
5. PunicaWrapper 如何把 LoRA id 转成 kernel 可用的 slot index？
6. Linear / embedding / logits LoRA layer 如何按 mapping 叠加 delta？
7. prefill / decode / spec decode 混合 batch 下 mapping 有什么区别？
8. mixed LoRA 的性能和限制在哪里？
```

阅读顺序建议：

```text
06_batch_mixed_lora_execution.md
  → 04_worker_model_runner_lora_state.md
  → 05_lora_layer_injection.md
  → 09_lora_and_parallelism.md
  → 10_lora_lifecycle_and_control.md
```

---

## 1. 一句话回答

batch mixed LoRA 的核心是：

```text
同一个 forward 中，
base model 仍然按整个 batch 统一计算，
LoRA layer 再根据 token_lora_mapping / prompt_lora_mapping，
为每个 token 或 sampled position 选择对应 LoRA slot，
计算低秩 delta 并加回 base output。
```

最小主线是：

```text
Request.lora_request
  → CachedRequestState.lora_request
  → InputBatch.add_request()
      → request_lora_mapping[req_index] = lora_int_id 或 0
      → lora_id_to_lora_request[lora_id] = LoRARequest
  → GPUModelRunner._prepare_inputs()
      → num_scheduled_tokens / num_sampled_tokens
      → set_active_loras(...)
  → InputBatch.make_lora_inputs()
      → token_lora_mapping = request_lora_mapping.repeat(num_scheduled_tokens)
      → prompt_lora_mapping = request_lora_mapping.repeat(num_sampled_tokens)
      → active_lora_requests
  → LoRAModelRunnerMixin._set_active_loras()
      → LoRAMapping(...)
      → lora_manager.set_active_adapters(...)
  → WorkerLoRAManager._apply_adapters()
      → load / activate LoRA if needed
  → LoRAModelManager.set_adapter_mapping()
      → PunicaWrapper.update_metadata(...)
  → LoRA layer forward
      → base output + selected LoRA delta
```

对应源码入口：

- `code/vllm/vllm/v1/worker/gpu_input_batch.py:468`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:976`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2193`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:73`
- `code/vllm/vllm/lora/worker_manager.py:183`
- `code/vllm/vllm/lora/model_manager.py:1139`
- `code/vllm/vllm/lora/punica_wrapper/punica_gpu.py:75`

---

## 2. batch mixed LoRA 要解决什么问题

### 2.1 一个 batch 内可以混合多种 LoRA 状态

例子：

```text
当前 persistent batch request 顺序：

req0: no LoRA
req1: LoRA A, lora_int_id = 10
req2: LoRA A, lora_int_id = 10
req3: LoRA B, lora_int_id = 20
req4: no LoRA
```

这一轮 scheduler 可能给每个 request 分配不同 token 数：

```text
num_scheduled_tokens:
  req0: 1    decode
  req1: 8    chunked prefill
  req2: 1    decode
  req3: 4    chunked prefill
  req4: 1    decode
```

所以 LoRA 执行不能只设置一个全局 adapter。

如果设置成“当前模型使用 LoRA A”，那么 req3 会错；如果设置成 LoRA B，req1/req2 会错；如果关闭 LoRA，LoRA 请求会错。

vLLM 的解决方式是：

```text
request 级选择：
  req_index -> lora_int_id

token 级展开：
  token_index -> lora_int_id

kernel 级转换：
  token_index -> lora_slot_index
```

### 2.2 no-LoRA 请求也是 mixed batch 的一部分

无 LoRA 请求不是被拆出去单独 forward，而是在 mapping 中用 `0` 表示。

在 `InputBatch.add_request()` 中：

```text
if request.lora_request:
    request_lora_mapping[req_index] = lora_id
else:
    request_lora_mapping[req_index] = 0
```

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:468`

后续 `convert_mapping()` 会把 `0` 转成 kernel 侧的 `-1`，表示该 token 不应用 LoRA delta。

源码位置：`code/vllm/vllm/lora/punica_wrapper/utils.py:95`

---

## 3. 第一层：InputBatch 保存 request 级 LoRA 状态

### 3.1 `CachedRequestState` 保存原始 LoRARequest

worker 侧每个请求会变成 `CachedRequestState`。

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:33`

其中 LoRA 字段是：

```text
lora_request: LoRARequest | None = None
```

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:50`

它表示这个 request 想使用哪个 adapter，但它还不是 token 级 mapping。

### 3.2 `InputBatch` 维护三个 LoRA 状态表

`InputBatch.__init__()` 初始化 LoRA 相关结构：

```text
request_lora_mapping: np.ndarray[max_num_reqs]
  req_index -> lora_int_id，0 表示 no-LoRA

lora_id_to_request_ids: dict[int, set[str]]
  lora_int_id -> 当前 batch 中使用它的 request ids

lora_id_to_lora_request: dict[int, LoRARequest]
  lora_int_id -> LoRARequest，用于后续加载 / 激活 adapter
```

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:244`

### 3.3 `add_request()` 写入 request 级 LoRA 映射

当新 request 加入 persistent batch 时，`InputBatch.add_request()` 会根据 `request.lora_request` 更新这三个结构。

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:335`

逻辑是：

```text
if request.lora_request:
    lora_id = request.lora_request.lora_int_id
    request_lora_mapping[req_index] = lora_id
    lora_id_to_request_ids[lora_id].add(req_id)
    lora_id_to_lora_request[lora_id] = request.lora_request
else:
    request_lora_mapping[req_index] = 0
```

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:468`

这一步只建立“request -> LoRA id”的关系。

### 3.4 `remove_request()` 清理 request 级 LoRA 状态

当 request 从 persistent batch 移除时，`remove_request()` 会同步清理 LoRA 状态。

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:510`

逻辑是：

```text
lora_id = request_lora_mapping[req_index]
if lora_id != 0:
    lora_id_to_request_ids[lora_id].discard(req_id)
    if no request uses this lora_id:
        delete lora_id_to_request_ids[lora_id]
        delete lora_id_to_lora_request[lora_id]
    request_lora_mapping[req_index] = 0
```

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:530`

注意：

```text
这里清理的是当前 batch 的 request 引用关系，
不是一定把 LoRA adapter 从 worker cache 中卸载。
```

adapter cache 的加载 / 淘汰由 worker LoRA manager 管。

---

## 4. 第二层：每轮执行前生成 token 级 mapping

### 4.1 `_prepare_inputs()` 末尾调用 `set_active_loras()`

`GPUModelRunner._prepare_inputs()` 完成本轮 token 布局、logits indices、spec decode metadata 之后，如果启用了 LoRA，会调用：

```text
self.set_active_loras(
    self.input_batch,
    num_scheduled_tokens,
    num_sampled_tokens,
)
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2193`

这里的两个数组含义不同：

```text
num_scheduled_tokens:
  每个 request 本轮送进 model forward 的 token 数。
  用于构造 token_lora_mapping。

num_sampled_tokens:
  每个 request 本轮需要采样 / 计算 logits 的 token 数。
  普通 decode 通常是 1；spec decode 时可能是 draft_len + 1。
  用于构造 prompt_lora_mapping。
```

### 4.2 `make_lora_inputs()` 执行 request -> token 展开

核心函数是 `InputBatch.make_lora_inputs()`。

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:976`

它返回三样东西：

```text
prompt_lora_mapping: tuple[int, ...]
  长度 = sum(num_sampled_tokens)
  第 i 个 sampled/logits 位置使用哪个 LoRA id。

token_lora_mapping: tuple[int, ...]
  长度 = sum(num_scheduled_tokens)
  第 i 个 input token 使用哪个 LoRA id。

active_lora_requests: set[LoRARequest]
  当前 batch 中涉及的 LoRARequest 集合。
```

源码注释位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:979`

实际实现非常关键：

```text
req_lora_mapping = request_lora_mapping[:num_reqs]
prompt_lora_mapping = req_lora_mapping.repeat(num_sampled_tokens)
token_lora_mapping = req_lora_mapping.repeat(num_scheduled_tokens)
active_lora_requests = set(lora_id_to_lora_request.values())
```

源码位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:991`

也就是说，mapping 的顺序完全依赖当前 batch 的 request 顺序和本轮 token 展开顺序。

### 4.3 一个具体例子

假设当前 batch：

```text
req index:  0        1        2        3        4
req id:     req0     req1     req2     req3     req4
LoRA id:    0        10       10       20       0
```

当前轮 token 分配：

```text
num_scheduled_tokens = [1, 8, 1, 4, 1]
num_sampled_tokens   = [1, 1, 1, 1, 1]
```

那么：

```text
request_lora_mapping[:5]
  = [0, 10, 10, 20, 0]

token_lora_mapping
  = [
      0,
      10,10,10,10,10,10,10,10,
      10,
      20,20,20,20,
      0,
    ]

prompt_lora_mapping
  = [0, 10, 10, 20, 0]
```

含义是：

```text
- req1 的 8 个 prefill token 都使用 LoRA A；
- req2 的 decode token 也使用 LoRA A；
- req3 的 4 个 token 使用 LoRA B；
- req0 / req4 不加 LoRA delta。
```

### 4.4 spec decode 下 `prompt_lora_mapping` 会变长

如果 spec decode 中某个 request 本轮要验证多个 draft token，并采样 bonus token，`num_sampled_tokens` 可能大于 1。

`GPUModelRunner._prepare_inputs()` 中：

```text
num_sampled_tokens = num_draft_tokens + 1
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2186`

此时 `prompt_lora_mapping` 会按 `num_sampled_tokens` 展开：

```text
req1: LoRA A, num_sampled_tokens=4
  → prompt_lora_mapping 里连续 4 个 LoRA A
```

它用于 logits processor / sampler 相关 LoRA 层，保证多个 sampled/logits position 仍然使用该 request 的 adapter。

---

## 5. 第三层：LoRAMapping 和 active adapter 设置

### 5.1 `LoRAModelRunnerMixin.set_active_loras()` 包装 mapping

`GPUModelRunner` 通过 `LoRAModelRunnerMixin` 获得 LoRA 能力。

`set_active_loras()` 调用 `input_batch.make_lora_inputs()`，然后进入 `_set_active_loras()`。

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:73`

`_set_active_loras()` 构造：

```text
LoRAMapping(
  index_mapping=token_lora_mapping,
  prompt_mapping=prompt_lora_mapping,
  is_prefill=True,
  type=LoRAMappingType.LANGUAGE,
)
```

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:48`

`LoRAMapping` 定义很简单：

```text
index_mapping: tuple[int, ...]
prompt_mapping: tuple[int, ...]
is_prefill: bool = False
type: LANGUAGE / TOWER / CONNECTOR
```

源码位置：`code/vllm/vllm/lora/layers/utils.py:27`

### 5.2 `is_prefill=True` 的含义

`_set_active_loras()` 总是把 `is_prefill` 设置成 `True`。

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:57`

源码注释说明：

```text
在非 CUDA 平台上总是使用 SGMV kernels；
在 CUDA 平台上 prefill / decode 使用同一套 kernels，该 flag 通常被忽略。
```

所以不要把这里的 `is_prefill=True` 理解成“当前 batch 一定全是 prefill”。它更多是 kernel 选择兼容字段。

### 5.3 active_lora_requests 用于保证 adapter 已加载

`LoRAModelRunnerMixin._set_active_loras()` 最终调用：

```text
self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
```

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:67`

其中 `lora_requests` 来自当前 `InputBatch.lora_id_to_lora_request.values()`。

它表示：

```text
当前 batch 中至少有一个 request 使用这些 LoRA；
worker manager 需要确保这些 adapter 已经加载并激活到 GPU slot。
```

---

## 6. 第四层：WorkerLoRAManager 加载和激活 adapter

### 6.1 `set_active_adapters()` 分两步

`WorkerLoRAManager.set_active_adapters()` 做两件事：

```text
1. _apply_adapters(requests)
   确保请求需要的 LoRA adapter 已加载 / 激活。

2. _adapter_manager.set_adapter_mapping(mapping)
   把本轮 token mapping 写入底层 LoRA model manager / PunicaWrapper。
```

源码位置：`code/vllm/vllm/lora/worker_manager.py:183`

### 6.2 V1 的 LRU LoRA manager 检查 GPU slot 数量

`LRUCacheWorkerLoRAManager._apply_adapters()` 会先构造：

```text
loras_map = {lora_int_id: LoRARequest}
```

然后检查：

```text
len(loras_map) <= self._adapter_manager.lora_slots
```

源码位置：`code/vllm/vllm/lora/worker_manager.py:258`

如果同一个 batch 中 distinct LoRA 数量超过 `max_loras`，会报错：

```text
Number of requested LoRAs (...) is greater than the number of GPU LoRA slots (...).
```

源码位置：`code/vllm/vllm/lora/worker_manager.py:264`

这就是 mixed LoRA 的第一个硬限制：

```text
同一个 batch 内可同时使用的 distinct LoRA adapter 数量不能超过 GPU LoRA slot 数。
```

### 6.3 adapter cache 和 active mapping 是两件事

`LRUCacheWorkerLoRAManager.add_adapter()` 负责加载 adapter 到 worker cache，并在需要时触发 LRU 淘汰。

源码位置：`code/vllm/vllm/lora/worker_manager.py:273`

`LoRAModelManager.activate_adapter()` 负责把 adapter 权重放进某个 GPU slot。

源码位置：`code/vllm/vllm/lora/model_manager.py:285`

它会：

```text
1. 找到第一个空的 lora_index_to_id slot；
2. lora_index_to_id[slot] = lora_model.id；
3. 遍历所有 LoRA-wrapped module；
4. module.set_lora(slot, lora_a, lora_b)。
```

源码位置：`code/vllm/vllm/lora/model_manager.py:292`

关键边界：

```text
adapter cache / GPU slot：
  决定某个 LoRA 权重是否已在 GPU 上。

LoRAMapping：
  决定当前 batch 的每个 token 应该读哪个 GPU slot。
```

---

## 7. 第五层：LoRAModelManager 把 id 映射到 GPU slot

### 7.1 `lora_index_to_id` 是 slot -> adapter id

`LoRAModelManager` 初始化时维护：

```text
lora_index_to_id: list[int | None] = [None] * lora_slots
```

源码位置：`code/vllm/vllm/lora/model_manager.py:103`

它的方向是：

```text
GPU slot index -> lora_int_id
```

例如：

```text
lora_index_to_id = [10, 20, None, None]

表示：
  slot 0 存 LoRA A, lora_int_id=10
  slot 1 存 LoRA B, lora_int_id=20
```

### 7.2 LoRAMapping 里保存的是 LoRA id，不是 slot index

`InputBatch.make_lora_inputs()` 生成的是：

```text
0 / 10 / 20 / ...
```

也就是 request 使用的 `lora_int_id`。

kernel 不能直接用这个 id 索引权重张量，因为权重张量按 GPU slot 排列。

所以需要 `convert_mapping()` 做转换。

### 7.3 `convert_mapping()` 把 LoRA id 转成 slot index

`PunicaWrapperBase._update_base_metadata()` 调用 `convert_mapping()`。

源码位置：`code/vllm/vllm/lora/punica_wrapper/punica_base.py:168`

`convert_mapping()` 的核心逻辑是：

```text
for x in mapping.index_mapping:
    if x > 0:
        lora_idx = lora_index_to_id.index(x)
    else:
        lora_idx = -1
```

源码位置：`code/vllm/vllm/lora/punica_wrapper/utils.py:99`

对 `prompt_mapping` 也是同样语义：

```text
lora_index_to_id.index(x) if x > 0 else -1
```

源码位置：`code/vllm/vllm/lora/punica_wrapper/utils.py:95`

因此：

```text
LoRA id 0：
  no-LoRA，kernel index = -1

LoRA id 10：
  如果 lora_index_to_id = [10, 20, None, None]
  kernel index = 0

LoRA id 20：
  kernel index = 1
```

### 7.4 `-1` 是 kernel 侧 no-LoRA 语义

`PunicaWrapperBase.token_lora_indices` 的注释明确说明：

```text
An index of -1 means no lora should be applied.
```

源码位置：`code/vllm/vllm/lora/punica_wrapper/punica_base.py:249`

所以 no-LoRA 请求仍然保留在 batch 中，只是在 LoRA kernel 中跳过 delta。

---

## 8. 第六层：PunicaWrapper 更新 kernel metadata

### 8.1 GPU wrapper 维护两套 metadata

`PunicaWrapperGPU` 初始化时创建：

```text
token_mapping_meta
prompt_mapping_meta
```

源码位置：`code/vllm/vllm/lora/punica_wrapper/punica_gpu.py:57`

两者分别对应：

```text
token_mapping_meta：
  用于普通 LoRA linear / embedding 计算，长度跟 input token 数相关。

prompt_mapping_meta：
  用于 logits processor / sampled positions，长度跟 sampled/logits position 数相关。
```

### 8.2 `update_metadata()` 将 mapping 写入 GPU tensor

`PunicaWrapperGPU.update_metadata()` 做三件事：

```text
1. self.is_prefill = mapping.is_prefill
2. _update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)
3. token_mapping_meta.prepare_tensors(self.token_lora_indices)
4. prompt_mapping_meta.prepare_tensors(self.sampler_indices)
```

源码位置：`code/vllm/vllm/lora/punica_wrapper/punica_gpu.py:75`

`_update_base_metadata()` 会把 CPU 侧 mapping 转成 GPU tensor：

```text
_token_lora_indices
_sampler_indices
_sampler_indices_padded
_embeddings_indices
```

源码位置：`code/vllm/vllm/lora/punica_wrapper/punica_base.py:168`

这些 tensor 是后续 Triton / custom op kernel 的索引依据。

### 8.3 `set_adapter_mapping()` 有 mapping 缓存

`LoRAModelManager.set_adapter_mapping()` 会比较 `_last_mapping`：

```text
if self._last_mapping != mapping:
    self._set_adapter_mapping(mapping)
    self._last_mapping = mapping
```

源码位置：`code/vllm/vllm/lora/model_manager.py:1139`

这避免在 mapping 未变化时重复准备 Punica metadata。

不过在实际在线服务中，batch request 组合和 token 数经常变化，所以 mixed LoRA mapping 仍然是一个每步可能变化的运行时状态。

---

## 9. 第七层：LoRA layer 如何按 mapping 叠加 delta

### 9.1 Linear LoRA 的基本公式

LoRA 对 Linear 的增量是：

```text
base_output = x @ W_base
lora_delta  = (x @ A_lora) @ B_lora * scale
output      = base_output + lora_delta
```

batch mixed LoRA 的区别是：

```text
每个 token 的 A_lora / B_lora 不是同一套，
而是根据 token_lora_indices[token_index] 选择对应 GPU slot。
```

### 9.2 `BaseLinearLayerWithLoRA` 先跑 base layer，再加 LoRA

Linear LoRA wrapper 的同步路径是：

```text
output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
return self._apply_lora_to_output(x, output)
```

源码位置：`code/vllm/vllm/lora/layers/base_linear.py:195`

`_apply_lora_to_output()` 调用：

```text
self.punica_wrapper.add_lora_linear(
    output,
    x,
    self.lora_a_stacked,
    self.lora_b_stacked,
    1.0,
    self.output_slices,
)
```

源码位置：`code/vllm/vllm/lora/layers/base_linear.py:206`

这里的 `output` 是整个 batch 的 base output，LoRA delta 会按 mapping 加回去。

### 9.3 Punica 的 linear 路径分 shrink 和 expand

`PunicaWrapperGPU.add_lora_linear()` 做两步：

```text
buffer = x @ A_lora       # shrink 到 rank 维度
output += buffer @ B_lora # expand 回 output hidden 维度
```

源码位置：`code/vllm/vllm/lora/punica_wrapper/punica_gpu.py:203`

内部调用：

```text
add_shrink(...)
add_expand(...)
```

源码位置：`code/vllm/vllm/lora/punica_wrapper/punica_gpu.py:250`

`add_shrink()` 和 `add_expand()` 都通过 `token_mapping_meta.meta_args(...)` 把 token 到 LoRA slot 的映射传给 kernel。

源码位置：

- `code/vllm/vllm/lora/punica_wrapper/punica_gpu.py:112`
- `code/vllm/vllm/lora/punica_wrapper/punica_gpu.py:158`

### 9.4 no-LoRA token 不加 delta

dual-stream 路径里有一段注释能说明 no-LoRA 的语义：

```text
_lora_expand_kernel exits early when lora_id == -1 (no active LoRA)
```

源码位置：`code/vllm/vllm/lora/layers/base_linear.py:247`

所以 no-LoRA token 的输出就是 base output，不会再叠加 delta。

### 9.5 Embedding LoRA 也使用同一套 mapping

`VocabParallelEmbeddingWithLoRA.forward()` 会取：

```text
indices_1 = self.punica_wrapper._embeddings_indices[1][:num_tokens]
```

然后用它偏移 embedding lookup：

```text
full_lora_a_embeddings = F.embedding(x + indices_1, self.lora_a_stacked_2d)
```

源码位置：`code/vllm/vllm/lora/layers/vocal_parallel_embedding.py:96`

之后再调用：

```text
punica_wrapper.add_lora_embedding(...)
```

源码位置：`code/vllm/vllm/lora/layers/vocal_parallel_embedding.py:119`

这说明 embedding LoRA 也不是按 request 拆 batch，而是通过映射后的 indices 在一个 batch 内选择不同 adapter。

### 9.6 Logits LoRA 使用 `prompt_mapping_meta`

`PunicaWrapperGPU.add_lora_logits()` 用的是 `prompt_mapping_meta`：

```text
lora_shrink(..., *self.prompt_mapping_meta.meta_args(...))
lora_expand(..., *self.prompt_mapping_meta.meta_args(...))
```

源码位置：`code/vllm/vllm/lora/punica_wrapper/punica_gpu.py:266`

这就是为什么 `prompt_lora_mapping` 要按 `num_sampled_tokens` 展开：logits / sampler 侧处理的不是所有输入 token，而是需要输出 logits 的 sampled positions。

---

## 10. prefill / decode / chunked prefill / spec decode 的区别

### 10.1 普通 decode

普通 decode 中，每个 request 通常本轮调度 1 个 token：

```text
num_scheduled_tokens = [1, 1, 1, ...]
num_sampled_tokens   = [1, 1, 1, ...]
```

此时：

```text
token_lora_mapping 约等于 request_lora_mapping
prompt_lora_mapping 也约等于 request_lora_mapping
```

### 10.2 chunked prefill

chunked prefill 中，一个 request 本轮可能调度多个 prompt token：

```text
num_scheduled_tokens = [128, 1, 64, ...]
```

这时 `token_lora_mapping` 会把该 request 的 LoRA id 重复多次：

```text
LoRA A request, 128 scheduled tokens
  → token_lora_mapping 里连续 128 个 A
```

但 `num_sampled_tokens` 可能仍然是 1，因为这一轮只需要一个 logits position。

### 10.3 prefill 和 decode 可以混在同一个 batch

vLLM V1 scheduler 并没有严格分离 prefill phase 和 decode phase，同一 batch 可以有：

```text
req0: decode 1 token, no LoRA
req1: prefill 128 tokens, LoRA A
req2: decode 1 token, LoRA B
```

LoRA mapping 仍然只按本轮 token 展开，不要求 batch 是纯 prefill 或纯 decode。

这就是 mixed LoRA 能和 chunked prefill 共存的原因。

### 10.4 spec decode

spec decode 中，`num_scheduled_tokens` 包含 target model 要验证的 draft tokens，`num_sampled_tokens` 是 `num_draft_tokens + 1`。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2164`

因此：

```text
token_lora_mapping：
  覆盖 target forward 的输入 tokens。

prompt_lora_mapping：
  覆盖多个 logits / sampled positions。
```

只要同一个 request 使用同一个 LoRARequest，它的 draft token 验证和最终采样位置都会映射到同一个 LoRA id。

---

## 11. 和 CUDA graph / warmup 的关系

### 11.1 LoRA 需要 dummy mapping 做 warmup / capture

`LoRAModelRunnerMixin` 提供 dummy LoRA 上下文：

```text
maybe_setup_dummy_loras()
maybe_select_dummy_loras()
maybe_dummy_run_with_lora()
```

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:93`

这些函数用于 warmup / CUDA graph capture 时构造模拟 mapping。

### 11.2 dummy mapping 可以模拟 active LoRA 数量

`maybe_select_dummy_loras()` 根据 `num_active_loras` 构造：

```text
prompt_lora_mapping
token_lora_mapping
dummy LoRARequest set
```

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:132`

它还支持一种特殊情况：

```text
num_active_loras > max_loras
  → 表示 max_loras 个 LoRA + no-LoRA tokens
```

源码位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:165`

这说明 CUDA graph / kernel specialization 需要区分 active LoRA 数量，以及是否混有 no-LoRA token。

### 11.3 Punica metadata 支持 active LoRA 数量 specialization

`PunicaWrapperGPU` 初始化时调用：

```text
get_captured_lora_counts(max_loras, specialize_active_lora)
LoRAKernelMeta.make(..., captured_lora_counts=...)
```

源码位置：`code/vllm/vllm/lora/punica_wrapper/punica_gpu.py:52`

这和性能有关：

```text
active LoRA 数量越动态，kernel metadata / graph capture 的组合越多；
如果 specialize_active_lora 开启，会按 active LoRA 数量做更多 specialization。
```

---

## 12. 性能边界

### 12.1 mixed LoRA 避免了多次模型 forward

如果不做 mixed LoRA，可能需要：

```text
base-only requests  → forward 1
LoRA A requests     → forward 2
LoRA B requests     → forward 3
```

vLLM 的 mixed LoRA 让它变成：

```text
所有请求同一个 batch：
  base forward 一次
  LoRA delta kernel 按 token mapping 叠加
```

这通常比按 LoRA 拆 batch 更好，因为保留了大 batch 的 attention / MLP 计算效率。

### 12.2 代价是 LoRA kernel 和 metadata 更复杂

mixed LoRA 的额外开销包括：

```text
- 每轮构造 token_lora_mapping / prompt_lora_mapping；
- LoRA id 到 GPU slot index 的转换；
- Punica metadata prepare_tensors；
- shrink / expand 两段 LoRA kernel；
- active adapter 数量变化导致的 kernel specialization / graph capture 压力。
```

### 12.3 distinct LoRA 数量越多，越接近最坏情况

同一个 batch 中 distinct LoRA 数量越多：

```text
- 需要更多 GPU LoRA slots；
- mapping 更分散；
- kernel 内按 LoRA 分组 / 索引的开销更高；
- CUDA graph specialization 组合更多；
- adapter cache 更容易触发 LRU 变化。
```

硬限制来自 `max_loras`：

```text
len(distinct_loras_in_batch) <= lora_slots
```

源码位置：`code/vllm/vllm/lora/worker_manager.py:264`

### 12.4 LoRA rank 越大，delta 计算越重

LoRA linear delta 的计算量近似来自：

```text
x @ A: hidden_size -> rank
rank @ B: rank -> output_size
```

因此 rank 越大，`lora_shrink` / `lora_expand` 的开销越高。

相关权重 buffer 在 `BaseLinearLayerWithLoRA.create_lora_weights()` 中按 `max_lora_rank` 分配。

源码位置：`code/vllm/vllm/lora/layers/base_linear.py:99`

### 12.5 no-LoRA token 仍参与 base forward

no-LoRA token 不跑 LoRA delta，但仍参与 base model forward。

这符合语义：

```text
LoRA 是 base model 之上的增量；
no-LoRA 请求就是只使用 base model。
```

---

## 13. 和多模态 tower / connector LoRA 的关系

本文主要讨论 language model 的 batch mixed LoRA。

多模态 tower / connector LoRA 也使用 `LoRAMapping`，但 mapping type 不同：

```text
LoRAMappingType.LANGUAGE
LoRAMappingType.TOWER
LoRAMappingType.CONNECTOR
```

定义位置：`code/vllm/vllm/lora/layers/utils.py:27`

`GPUModelRunner._execute_mm_encoder()` 在处理多模态 encoder batch 时，会为 tower / connector 单独构造 mapping。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2941`

原因是：

```text
language model batch：
  按 request/token 组织。

multimodal encoder batch：
  按 multimodal item / encoder tokens 组织。
```

所以多模态 LoRA 不能直接复用 language model 的 `token_lora_mapping`，需要按 encoder item 重新构造。

---

## 14. 和 MoE / Expert Parallel 的关系

MoE LoRA 也需要按 token 找到 LoRA slot，但还多了 expert 维度。

`PunicaWrapperGPU.moe_lora_align_block_size()` 支持传入 `token_lora_mapping`，并说明：

```text
When token_lora_mapping is provided, it overrides the global mapping read from self.token_mapping_meta.
This is how EP+LoRA injects the per-rank-local token→LoRA map after all-to-all dispatch.
```

源码位置：`code/vllm/vllm/lora/punica_wrapper/punica_gpu.py:327`

这说明在 EP 场景下：

```text
全局 batch token mapping
  → all-to-all / expert dispatch
  → 每个 rank 本地 token mapping
  → MoE LoRA kernel
```

因此 EP+LoRA 的 mapping 可能不再直接使用最初的全局 `token_lora_mapping`，而是使用 dispatch 后的本地 token->LoRA 映射。

---

## 15. 常见误区

### 15.1 “active adapter” 不是“当前全局唯一 adapter”

在 mixed LoRA 中，active adapters 是一个集合：

```text
{LoRA A, LoRA B, ...}
```

真正决定某个 token 用哪个 adapter 的是 mapping：

```text
token_lora_mapping[token_index]
```

### 15.2 `request_lora_mapping` 不是 kernel 可直接使用的 mapping

`request_lora_mapping` 是 request 级：

```text
req_index -> lora_int_id
```

kernel 需要的是 token 级 slot index：

```text
token_index -> lora_slot_index / -1
```

中间必须经过：

```text
make_lora_inputs()
  → LoRAMapping
  → convert_mapping()
  → Punica metadata
```

### 15.3 LoRA id 0 和 slot 0 不是一回事

`0` 在 request / mapping 层表示 no-LoRA。

slot `0` 在 kernel / weight 层可能表示第一个真实 LoRA adapter。

转换例子：

```text
request mapping: [0, 10, 20]
lora_index_to_id: [10, 20]
kernel mapping:  [-1, 0, 1]
```

### 15.4 相同 LoRA 的多个 request 共享同一套 GPU 权重

如果 req1 和 req2 都使用 LoRA A：

```text
request_lora_mapping = [..., 10, 10, ...]
```

它们最终都会映射到同一个 slot，比如 slot 0。

不会重复加载两份 LoRA A 权重。

### 15.5 mixed LoRA 不等于把 LoRA 权重合并进 base model

vLLM 的动态 LoRA 不会为每个 batch 修改 base model 权重。

它是：

```text
base output + runtime LoRA delta
```

adapter 权重常驻在 LoRA slot tensor 中，mapping 决定本轮哪些 token 使用哪些 slot。

---

## 16. 主链路总表

```text
请求进入：
  LoRARequest
    → Request.lora_request
    → CachedRequestState.lora_request

batch 状态：
  InputBatch.add_request()
    → request_lora_mapping[req_index]
    → lora_id_to_request_ids
    → lora_id_to_lora_request

每轮执行：
  GPUModelRunner._prepare_inputs()
    → num_scheduled_tokens
    → num_sampled_tokens
    → set_active_loras()

mapping 展开：
  InputBatch.make_lora_inputs()
    → token_lora_mapping
    → prompt_lora_mapping
    → active_lora_requests

adapter 准备：
  WorkerLoRAManager.set_active_adapters()
    → _apply_adapters()
    → add_adapter() / activate_adapter()

slot 映射：
  LoRAModelManager.lora_index_to_id
    → convert_mapping()
    → token_lora_indices / sampler_indices

kernel metadata：
  PunicaWrapper.update_metadata()
    → token_mapping_meta.prepare_tensors()
    → prompt_mapping_meta.prepare_tensors()

layer forward：
  base layer output
    → punica_wrapper.add_lora_linear / embedding / logits
    → selected LoRA delta
    → output
```

---

## 17. 推荐源码阅读路线

### 17.1 request 级 LoRA 状态

```text
vllm/v1/worker/gpu_input_batch.py
  → CachedRequestState
  → InputBatch.__init__()
  → InputBatch.add_request()
  → InputBatch.remove_request()
```

### 17.2 token 级 mapping

```text
vllm/v1/worker/gpu_model_runner.py
  → _prepare_inputs()
  → set_active_loras(...)

vllm/v1/worker/lora_model_runner_mixin.py
  → set_active_loras()
  → _set_active_loras()

vllm/v1/worker/gpu_input_batch.py
  → make_lora_inputs()
```

### 17.3 adapter 加载和 slot 映射

```text
vllm/lora/worker_manager.py
  → set_active_adapters()
  → _apply_adapters()
  → add_adapter()

vllm/lora/model_manager.py
  → activate_adapter()
  → set_adapter_mapping()
  → _set_adapter_mapping()
```

### 17.4 kernel metadata 和 layer 执行

```text
vllm/lora/punica_wrapper/utils.py
  → convert_mapping()

vllm/lora/punica_wrapper/punica_base.py
  → _update_base_metadata()

vllm/lora/punica_wrapper/punica_gpu.py
  → update_metadata()
  → add_lora_linear()
  → add_lora_logits()

vllm/lora/layers/base_linear.py
  → _apply_lora_to_output()

vllm/lora/layers/vocal_parallel_embedding.py
  → forward()
```

---

## 18. 一句话总结

batch mixed LoRA 的本质是三次映射转换：

```text
request 级：
  req_index -> lora_int_id

token 级：
  token_index -> lora_int_id

kernel 级：
  token_index -> lora_slot_index 或 -1(no-LoRA)
```

vLLM 依靠这三层映射，让同一 batch 中的 no-LoRA、LoRA A、LoRA B 请求共用一次 base model forward，并在 LoRA-wrapped layer 中按 token 叠加各自 adapter 的低秩 delta。
