# 09. LoRA 如何与并行机制交互？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\model_manager.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\lora_model.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\layers\base_linear.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\layers\column_parallel_linear.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\layers\row_parallel_linear.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\layers\replicated_linear.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\layers\vocal_parallel_embedding.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\worker_manager.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\executor\abstract.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_input_batch.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\lora_model_runner_mixin.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`

本问题关注：在 TP / PP / DP / EP / 多模态 tower-connector LoRA 场景下，LoRA adapter 权重如何加载到正确 worker，如何按照 base layer 的并行切分方式切分或复制，以及当前 batch 的 active LoRA mapping 如何在各 rank 上保持一致。

---

## 1. 一句话回答

vLLM 的 LoRA 并行原则是：

```text
LoRA wrapper 必须匹配 base layer 的并行方式；
LoRA 权重在每个 worker 上按该 worker 持有的 base layer 分片写入本地 LoRA slot；
每轮 batch 的 LoRA mapping 由 Scheduler 约束、Worker 本地生成，并在参与同一个模型并行组的 ranks 上保持同一请求顺序和同一 lora_int_id 语义。
```

主链路是：

```text
load model with LoRA enabled
  → LoRAModelRunnerMixin.load_lora_model()
  → LRUCacheWorkerLoRAManager.create_lora_manager()
  → LoRAModelManager._create_lora_modules()
      → 跳过 PPMissingLayer
      → from_layer() 选择匹配并行层的 LoRA wrapper
      → create_lora_weights() 预分配 max_loras 个 LoRA slot
  → add_lora / set_active_loras
      → WorkerLoRAManager._load_adapter()
      → LoRAModel.from_local_checkpoint()
      → LoRAModelManager.activate_adapter()
      → module.set_lora(index, lora_a, lora_b)
      → wrapper.slice_lora_a / slice_lora_b 按 TP / EP / packed layout 写入本 rank
  → forward
      → base layer parallel forward
      → punica wrapper 按 LoRAMapping 加 LoRA delta
      → 必要时 all_gather / all_reduce
```

所以：

```text
Scheduler 不切 LoRA 权重；
Executor 负责把控制面广播到所有 worker；
LoRA layer wrapper 负责按本地 rank 的并行分片写入和执行。
```

---

## 2. LoRA 并行的三个层次

LoRA 和并行机制交互可以分三层看：

```text
1. 控制面同步：
   add_lora / remove_lora / pin_lora / list_loras 通过 Executor.collective_rpc 到所有 worker。

2. 权重布局：
   每个 worker 加载 adapter 后，只把自己负责的 LoRA A/B 分片写入本地 LoRA wrapper。

3. 执行映射：
   InputBatch 根据当前 batch 的 request_lora_mapping 生成 token_lora_mapping，
   LoRA manager 把 mapping 下发给 Punica wrapper，forward 时按 token 选择 adapter。
```

这三层分别对应：

| 层次 | 关键对象 | 源码位置 |
|---|---|---|
| 控制面同步 | `Executor.add_lora()` / `collective_rpc()` | `abstract.py:292` 到 `abstract.py:308` |
| 权重布局 | `BaseLayerWithLoRA.set_lora()` / `slice_lora_a()` / `slice_lora_b()` | `base_linear.py:157` 到 `base_linear.py:183` |
| 执行映射 | `InputBatch.make_lora_inputs()` / `LoRAMapping` | `gpu_input_batch.py:976` 到 `gpu_input_batch.py:999` |

---

## 3. LoRA wrapper 如何匹配并行层

LoRA wrapper 的选择入口是：`utils.py:106`

```python
def from_layer(
    layer: nn.Module,
    max_loras: int,
    lora_config: LoRAConfig,
    packed_modules_list: list,
    model_config: PretrainedConfig | None = None,
) -> nn.Module:
    for lora_cls in _all_lora_classes:
        if lora_cls.can_replace_layer(...):
            instance_layer = lora_cls(layer)
            instance_layer.create_lora_weights(max_loras, lora_config, model_config)
            return instance_layer
    return layer
```

位置：`utils.py:106` 到 `utils.py:124`

候选 wrapper 顺序定义在：`utils.py:78` 到 `utils.py:95`

```python
_all_lora_classes = (
    VocabParallelEmbeddingWithLoRA,
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    LogitsProcessorWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
    QKVParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearVariableSliceWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    RowParallelLinearWithShardedLoRA,
    FusedMoEWithLoRA,
    FusedMoE3DWithLoRA,
)
```

注释强调：

```text
Order matters：更具体的 wrapper 要先于通用 merged / column parallel wrapper 检查。
```

原因是 QKV、MergedColumn、MoE、VocabParallelEmbedding 都有特殊分片或 packed layout，不能简单用普通 Linear wrapper。

---

## 4. LoRAModelManager 如何创建并行 LoRA 模块

`LoRAModelManager` 初始化时会调用：

```python
self._init_punica_wrapper(max_num_batched_tokens, vllm_config)
self._create_lora_modules()
self.moe_ep_load_spec = self._build_moe_ep_load_spec()
```

位置：`model_manager.py:129` 到 `model_manager.py:135`

### 4.1 遍历模型模块并替换

`_create_lora_modules()` 定义在：`model_manager.py:375`

关键流程：

```python
for module_name, module in self.model.named_modules(remove_duplicate=False):
    if isinstance(module, PPMissingLayer):
        continue

    if not self._match_target_modules(module_name):
        continue

    punica_wrapper = self._get_punica_wrapper(module_name)
    if punica_wrapper is None:
        continue

    packed_moduled_lst = self.packed_modules_mapping.get(parts, [])
    if isinstance(module, MoERunner):
        packed_moduled_lst = ["w13"] if self._is_3d_moe_model else ["w1", "w3"]

    new_module = replace_submodule(
        self.model,
        module_name,
        from_layer(...),
    )
```

位置：`model_manager.py:383` 到 `model_manager.py:451`

几个并行相关点：

```text
- PPMissingLayer 会被跳过，说明当前 PP rank 没有这层，不创建 LoRA wrapper；
- from_layer() 根据当前 rank 上真实 base layer 类型选择 wrapper；
- MoERunner 会根据 2D/3D MoE 权重格式选择 FusedMoEWithLoRA 或 FusedMoE3DWithLoRA；
- 每个 wrapper 会复用同一个对应 prefix 的 PunicaWrapper。
```

### 4.2 wrapper 注册

替换成功后：

```python
self.register_module(module_name, new_module)
self._register_packed_modules(module_name)
new_module.set_mapping(punica_wrapper)
```

位置：`model_manager.py:497` 到 `model_manager.py:501`

`self.modules` 记录的是：

```text
module_name -> BaseLayerWithLoRA
```

后续 `activate_adapter()` 会遍历这些模块，把 LoRA 权重写入每个 wrapper 的对应 slot。

---

## 5. Tensor Parallel：ColumnParallelLinear

`ColumnParallelLinearWithLoRA` 定义在：`column_parallel_linear.py:83`

源码注释说：

```text
LoRA on top of ColumnParallelLinear layer.
LoRA B is sliced for tensor parallelism.
```

位置：`column_parallel_linear.py:83` 到 `column_parallel_linear.py:90`

### 5.1 普通 ColumnParallelLinear

普通 column parallel 的 base layer 输出维度按 TP 切分，所以 LoRA B 也按输出维度切分；LoRA A 不切。

```python
def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
    return lora_a

def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
    shard_size = self.output_size
    start_idx = self.tp_rank * shard_size
    end_idx = (self.tp_rank + 1) * shard_size
    lora_b = lora_b[start_idx:end_idx, :]
    return lora_b
```

位置：`column_parallel_linear.py:102` 到 `column_parallel_linear.py:128`

执行时：

```python
output_parallel = self.apply(input_, bias)
if self.base_layer.gather_output and self.tp_size > 1:
    output = tensor_model_parallel_all_gather(output_parallel)
else:
    output = output_parallel
```

位置：`column_parallel_linear.py:142` 到 `column_parallel_linear.py:150`

含义：

```text
本 rank 只产生自己负责的输出列；
如果 base layer 需要 gather_output，则 LoRA 后的 output_parallel 一起 all_gather。
```

### 5.2 MergedColumnParallelLinear

Merged column parallel 常见于 `gate_proj + up_proj -> gate_up_proj`。

对于两个 slice 的 merged layer：

```python
if self.is_merged_col_linear:
    shard_size = self.output_size // 2
    offset = lora_b.shape[0] // 2

    left_weight = lora_b[
        self.tp_rank * shard_size : (self.tp_rank + 1) * shard_size, :
    ]
    right_weight = lora_b[
        offset + self.tp_rank * shard_size : offset
        + (self.tp_rank + 1) * shard_size,
        :,
    ]
    lora_b = torch.cat([left_weight, right_weight], dim=0)
```

位置：`column_parallel_linear.py:105` 到 `column_parallel_linear.py:120`

也就是说：

```text
每个 TP rank 分别取 gate/up 两段中属于自己的输出 shard，再拼成本地 LoRA B。
```

---

## 6. Tensor Parallel：QKVParallelLinear

QKV fused layer 有特殊处理，因为 Q、K、V 的 head 数和分片方式可能不同。

`QKVParallelLinearWithLoRA` 定义在：`column_parallel_linear.py:365`

关键切片：

```python
self.q_shard_id = self.tp_rank
self.kv_shard_id = self.tp_rank // self.base_layer.num_kv_head_replicas

lora_b_q = lora_b[
    self.q_proj_shard_size * self.q_shard_id : self.q_proj_shard_size
    * (self.q_shard_id + 1),
    :,
]

k_offset = self.q_proj_total_size
lora_b_k = lora_b[
    k_offset + self.kv_proj_shard_size * self.kv_shard_id : k_offset
    + self.kv_proj_shard_size * (self.kv_shard_id + 1),
    :,
]

v_offset = k_offset + self.kv_proj_total_size
lora_b_v = lora_b[
    v_offset + self.kv_proj_shard_size * self.kv_shard_id : v_offset
    + self.kv_proj_shard_size * (self.kv_shard_id + 1),
    :,
]

lora_b = torch.cat([lora_b_q, lora_b_k, lora_b_v], dim=0)
```

位置：`column_parallel_linear.py:393` 到 `column_parallel_linear.py:414`

含义：

```text
Q 按 q head shard 切；
K/V 按 kv head shard 切；
GQA/MQA 场景下多个 TP rank 可能复用同一个 kv_shard_id；
最后把本 rank 的 q/k/v LoRA B 拼成 fused 本地权重。
```

对于 checkpoint 中 Q/K/V 分成三个 LoRA 的情况，使用 `MergedQKVParallelLinearWithLoRA`：`column_parallel_linear.py:431`

它设置：

```python
self.output_slices = (
    self.q_proj_shard_size,
    self.kv_proj_shard_size,
    self.kv_proj_shard_size,
)
self.output_ids = (
    self.q_shard_id,
    self.kv_shard_id,
    self.kv_shard_id,
)
```

位置：`column_parallel_linear.py:454` 到 `column_parallel_linear.py:463`

---

## 7. Tensor Parallel：RowParallelLinear

`RowParallelLinearWithLoRA` 定义在：`row_parallel_linear.py:22`

RowParallelLinear 的 base layer 输入维度按 TP 切分，所以 LoRA A 按输入维度切分；LoRA B 不切。

```python
def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
    shard_size = self.input_size
    start_idx = self.tp_rank * shard_size
    end_idx = (self.tp_rank + 1) * shard_size
    lora_a = lora_a[:, start_idx:end_idx]
    return lora_a

def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
    return lora_b
```

位置：`row_parallel_linear.py:32` 到 `row_parallel_linear.py:40`

forward 路径：

```python
if self.base_layer.input_is_parallel:
    input_parallel = input_
else:
    split_input = split_tensor_along_last_dim(
        input_, num_partitions=self.tp_size
    )
    input_parallel = split_input[self.tp_rank].contiguous()

output_parallel = self.apply(input_parallel, bias_)
if self.base_layer.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
else:
    output = output_parallel
```

位置：`row_parallel_linear.py:57` 到 `row_parallel_linear.py:76`

含义：

```text
每个 TP rank 只处理自己那段 input hidden；
LoRA delta 和 base output 一样先是 partial sum；
如果 reduce_results=True，最后 all_reduce 得到完整输出。
```

---

## 8. fully_sharded_loras 的 S-LoRA 路径

`LoRAConfig.fully_sharded_loras=True` 时，vLLM 使用 S-LoRA 风格的 fully sharded wrapper。

基础 shape 在 `BaseLinearLayerWithLoRA.create_lora_weights()` 中决定：`base_linear.py:99`

ColumnParallelLinear：

```python
lora_a_out_size = (
    lora_config.max_lora_rank
    if not lora_config.fully_sharded_loras
    else divide(lora_config.max_lora_rank, self.tp_size)
)
lora_b_out_size = self.output_size
```

位置：`base_linear.py:110` 到 `base_linear.py:117`

RowParallelLinear：

```python
lora_a_out_size = lora_config.max_lora_rank
lora_b_out_size = (
    self.output_size
    if not lora_config.fully_sharded_loras
    else divide(self.output_size, self.tp_size)
)
```

位置：`base_linear.py:118` 到 `base_linear.py:124`

### 8.1 ColumnParallel fully sharded

`ColumnParallelLinearWithShardedLoRA` 定义在：`column_parallel_linear.py:497`

它和普通 ColumnParallel 的区别是 LoRA A 也按 rank 维切分：

```python
def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
    shard_size = self.lora_a_stacked[0].shape[2]
    start_idx = self.tp_rank * shard_size
    lora_a = lora_a[start_idx : start_idx + shard_size, :]
    return lora_a
```

位置：`column_parallel_linear.py:504` 到 `column_parallel_linear.py:513`

执行时 `_mcp_apply()` 会：

```text
x @ local_lora_a -> local rank buffer
all_gather rank buffer
用本地 lora_b expand 到本地 output shard
```

关键 all_gather：

```python
buffers = tensor_model_parallel_all_gather(buffers)
```

位置：`column_parallel_linear.py:57` 到 `column_parallel_linear.py:66`

### 8.2 RowParallel fully sharded

`RowParallelLinearWithShardedLoRA` 定义在：`row_parallel_linear.py:101`

它和普通 RowParallel 的区别是 LoRA B 也按输出维切分：

```python
def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
    shard_size = self.lora_b_stacked[0].shape[2]
    start_idx = self.tp_rank * shard_size
    end_idx = (self.tp_rank + 1) * shard_size
    lora_b = lora_b[start_idx:end_idx, :]
    return lora_b
```

位置：`row_parallel_linear.py:111` 到 `row_parallel_linear.py:116`

执行时先对 shrink buffer 做 all_reduce：

```python
if self.tp_size > 1:
    buffer = tensor_model_parallel_all_reduce(buffer)
```

位置：`row_parallel_linear.py:129` 到 `row_parallel_linear.py:136`

然后只把 LoRA output 加到当前 rank 对应的 output slice：

```python
shard_size = self.lora_b_stacked[0].shape[2]
offset_start = self.tp_rank * shard_size
...
self.punica_wrapper.add_expand(..., offset_start=offset_start, add_input=True)
```

位置：`row_parallel_linear.py:137` 到 `row_parallel_linear.py:153`

源码注释说明：

```text
最终 output 不是普通 row_parallel 的完整输出，后续还需要标准 all_reduce。
```

位置：`row_parallel_linear.py:137` 到 `row_parallel_linear.py:142`

---

## 9. ReplicatedLinear 与 VocabParallelEmbedding

### 9.1 ReplicatedLinear

`ReplicatedLinearWithLoRA` 定义在：`replicated_linear.py:16`

ReplicatedLinear 在每个 GPU 上都有完整权重，所以 LoRA A/B 都不切：

```python
def slice_lora_a(...):
    return lora_a

def slice_lora_b(...):
    return lora_b
```

位置：`replicated_linear.py:67` 到 `replicated_linear.py:77`

注释说明：

```text
ReplicatedLinear should always be replaced, regardless of fully_sharded_loras setting,
because it is, by definition, copied per GPU.
```

位置：`replicated_linear.py:55` 到 `replicated_linear.py:65`

### 9.2 VocabParallelEmbedding

`VocabParallelEmbeddingWithLoRA` 定义在：`vocal_parallel_embedding.py:17`

它不是普通 A/B linear 形态，而是：

```python
self.lora_a_stacked = torch.zeros(
    (max_loras, self.base_layer.org_vocab_size, lora_config.max_lora_rank),
    ...
)
self.lora_b_stacked = torch.zeros(
    (max_loras, 1, self.base_layer.embedding_dim, lora_config.max_lora_rank),
    ...
)
```

位置：`vocal_parallel_embedding.py:49` 到 `vocal_parallel_embedding.py:67`

写入时会转置 LoRA A：

```python
self.lora_a_stacked[index, : lora_a.shape[1], : lora_a.shape[0]].copy_(
    lora_a.T, non_blocking=True
)
```

位置：`vocal_parallel_embedding.py:86` 到 `vocal_parallel_embedding.py:94`

forward 时先查 LoRA A embedding，再通过 Punica 加 embedding delta：

```python
full_lora_a_embeddings = F.embedding(
    x + indices_1,
    self.lora_a_stacked_2d,
)
full_output = self.base_layer.forward(x)
...
self.punica_wrapper.add_lora_embedding(
    full_output, full_lora_a_embeddings, self.lora_b_stacked, add_input=True
)
```

位置：`vocal_parallel_embedding.py:96` 到 `vocal_parallel_embedding.py:126`

---

## 10. Pipeline Parallel：每个 PP rank 只包装本地层

PP 相关最关键的源码点在 `LoRAModelManager._create_lora_modules()`：

```python
for module_name, module in self.model.named_modules(remove_duplicate=False):
    if isinstance(module, PPMissingLayer):
        continue
```

位置：`model_manager.py:383` 到 `model_manager.py:387`

含义：

```text
Pipeline parallel 下，每个 PP rank 的模型对象只包含自己负责执行的层；
不属于当前 PP rank 的层会以 PPMissingLayer 形式出现或不存在；
LoRA manager 不会为这些 missing layer 创建 LoRA wrapper，也不会加载这些层的 LoRA 权重到本 rank 的 GPU slot。
```

请求级 `LoRARequest` 仍然会随 `SchedulerOutput` 到达参与执行的 worker。每个 PP rank 在自己的 `InputBatch` 中维护同样的 request LoRA id 语义：

```text
req_id -> lora_int_id
```

但真正写入 GPU 的 LoRA 权重只覆盖本 PP rank 上存在的 modules。

PP 中 intermediate tensors 不携带 LoRA 权重。LoRA delta 已经在各 PP stage 的本地 forward 中叠加到该 stage 的 hidden states 上，随后 hidden states 作为普通 pipeline intermediate tensor 传递给下一 stage。

---

## 11. Data Parallel：每个 replica 都要有相同 adapter 语义

V1 executor 的 LoRA 控制接口在 `Executor` 抽象类里统一实现：

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
```

位置：`abstract.py:292` 到 `abstract.py:302`

`list_loras()` 会检查所有 worker 返回一致：

```python
sets: list[set[int]] = self.collective_rpc("list_loras")
for s in sets:
    assert s == sets[0], "All workers should have the same LORAs."
return sets[0]
```

位置：`abstract.py:304` 到 `abstract.py:308`

这说明 LoRA 控制面假设：

```text
一个 executor 管辖的所有 worker，已注册 / 已加载 LoRA 集合应保持一致。
```

DP 场景下，请求会被路由到某个 DP replica；该 replica 内部的 TP/PP worker 必须理解同一个 `lora_int_id`。因此：

```text
- add_lora/remove_lora/pin_lora 应同步到对应 DP replica 的 workers；
- 请求只在被路由到的 DP rank 上参与 batch；
- active mapping 是该 DP replica 本地 batch 的 mapping；
- 不同 DP replica 可以同时处理不同 batch，但相同 lora_int_id 应指向同一个 adapter 语义。
```

源码没有给 LoRA 单独实现 DP 分支，它依赖 Engine / Executor / collective_rpc 的并行 worker 管理机制来保证一致性。

---

## 12. Expert Parallel / MoE LoRA

MoE LoRA 的关键对象是 `MoERunner`、`FusedMoEWithLoRA`、`FusedMoE3DWithLoRA` 和 `MoEEPLoadSpec`。

`LoRAModelManager` 初始化时记录是否启用 EP：

```python
self._use_ep = bool(
    vllm_config and vllm_config.parallel_config.enable_expert_parallel
)
```

位置：`model_manager.py:128` 到 `model_manager.py:131`

如果模块是 `MoERunner`，会根据 MoE LoRA 权重格式选择 packed modules：

```python
if isinstance(module, MoERunner):
    packed_moduled_lst = ["w13"] if self._is_3d_moe_model else ["w1", "w3"]
```

位置：`model_manager.py:434` 到 `model_manager.py:440`

### 12.1 EP load spec

`MoEEPLoadSpec` 定义在：`lora_model.py:25`

```python
@dataclass(frozen=True)
class MoEEPLoadSpec:
    ep_rank: int
    local_num_experts: int
    global_num_experts: int
```

位置：`lora_model.py:25` 到 `lora_model.py:35`

注释说明：

```text
Per-expert-parallel slicing metadata for one FusedMoE LoRA module.
Threaded into the LoRA loader so per-expert weights from EP ranks other than this one can be skipped before they ever hit CPU memory.
```

位置：`lora_model.py:26` 到 `lora_model.py:30`

`LoRAModelManager._build_moe_ep_load_spec()` 在启用 EP 且模型是 MoE 时构造这个 spec。

位置：`model_manager.py:1084` 到 `model_manager.py:1105`

### 12.2 加载时跳过远端 experts

`LoRAModel.from_local_checkpoint()` 接收：

```python
moe_ep_spec: MoEEPLoadSpec | None = None
```

位置：`lora_model.py:166` 到 `lora_model.py:181`

对于 safetensors：

```python
for module in f.keys():
    if moe_ep_spec is not None and _is_remote_expert_key(
        module, moe_ep_spec
    ):
        continue
    tensors[module] = f.get_tensor(module)
```

位置：`lora_model.py:268` 到 `lora_model.py:276`

对于 bin / pt：

```python
if moe_ep_spec is not None:
    tensors = {
        k: v
        for k, v in tensors.items()
        if not _is_remote_expert_key(k, moe_ep_spec)
    }
```

位置：`lora_model.py:283` 到 `lora_model.py:293`

`_is_remote_expert_key()` 根据 expert index 判断是否属于当前 EP rank：

```python
local_start = spec.ep_rank * spec.local_num_experts
return not (local_start <= expert_idx < local_start + spec.local_num_experts)
```

位置：`lora_model.py:41` 到 `lora_model.py:57`

含义：

```text
EP rank 只加载本地专家的 LoRA 权重；
非本地 expert 的 LoRA tensor 在读取 checkpoint 时就跳过，避免不必要的 CPU/GPU 内存占用。
```

---

## 13. active LoRA mapping 如何在并行 rank 上一致

每个请求进入 Worker 后，`InputBatch.add_request()` 会记录：

```python
if request.lora_request:
    lora_id = request.lora_request.lora_int_id
    self.request_lora_mapping[req_index] = lora_id
    self.lora_id_to_request_ids[lora_id].add(request.req_id)
    self.lora_id_to_lora_request[lora_id] = request.lora_request
else:
    self.request_lora_mapping[req_index] = 0
```

位置：`gpu_input_batch.py:468` 到 `gpu_input_batch.py:479`

执行前生成 token 级 mapping：

```python
req_lora_mapping = self.request_lora_mapping[: self.num_reqs]
prompt_lora_mapping = tuple(req_lora_mapping.repeat(num_sampled_tokens))
token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))
active_lora_requests = set(self.lora_id_to_lora_request.values())
```

位置：`gpu_input_batch.py:991` 到 `gpu_input_batch.py:999`

`LoRAModelRunnerMixin.set_active_loras()` 把这些映射交给 LoRA manager：

```python
prompt_lora_mapping, token_lora_mapping, lora_requests = (
    input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)
)
return self._set_active_loras(
    prompt_lora_mapping, token_lora_mapping, lora_requests, mapping_type
)
```

位置：`lora_model_runner_mixin.py:73` 到 `lora_model_runner_mixin.py:91`

最后：

```python
lora_mapping = LoRAMapping(
    token_lora_mapping,
    prompt_lora_mapping,
    is_prefill=True,
    type=mapping_type,
)
self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
```

位置：`lora_model_runner_mixin.py:48` 到 `lora_model_runner_mixin.py:68`

并行一致性来自两个前提：

```text
1. SchedulerOutput 对同一并行执行组给出同一批 req_id / num_scheduled_tokens；
2. 所有 worker 对同一个 lora_int_id 通过控制面加载同一 adapter，并在本地 wrapper 中写入本 rank 对应分片。
```

---

## 14. Scheduler 对 max_loras 的约束

Scheduler 不切权重，但会避免一个 batch 中活跃 LoRA 数超过配置上限。

RUNNING 请求调度后记录本轮已用 LoRA：

```python
scheduled_loras = set(
    req.lora_request.lora_int_id
    for req in scheduled_running_reqs
    if req.lora_request and req.lora_request.lora_int_id > 0
)
assert len(scheduled_loras) <= self.lora_config.max_loras
```

位置：`scheduler.py:614` 到 `scheduler.py:622`

WAITING 请求进入本轮 batch 前检查：

```python
if (
    self.lora_config
    and request.lora_request
    and (
        len(scheduled_loras) == self.lora_config.max_loras
        and request.lora_request.lora_int_id not in scheduled_loras
    )
):
    # Scheduling would exceed max_loras, skip.
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

位置：`scheduler.py:651` 到 `scheduler.py:664`

请求成功加入 batch 后：

```python
if self.lora_config and request.lora_request:
    scheduled_loras.add(request.lora_request.lora_int_id)
```

位置：`scheduler.py:951` 到 `scheduler.py:952`

这保证：

```text
单个 scheduler step 内，Worker 需要同时激活的 LoRA 数不会超过 max_loras / GPU LoRA slots。
```

---

## 15. 多模态 tower / connector LoRA 的并行点

多模态模型可能有三类 LoRA mapping：

```text
LANGUAGE：语言模型主体；
TOWER：多模态 encoder tower；
CONNECTOR：多模态 connector。
```

`LoRAModelManager._maybe_init_mm()` 会为 language model、tower model、connector 创建不同的 PunicaWrapper：

```python
self.punica_wrapper_mapping[lm_prefix] = llm_punica_wrapper
...
for prefix in self.mm_mapping.tower_model:
    self.punica_wrapper_mapping[prefix] = tower_punica_wrapper
...
for prefix in self.mm_mapping.connector:
    self.punica_wrapper_mapping[prefix] = connector_punica_wrapper
```

位置：`model_manager.py:177` 到 `model_manager.py:262`

`_set_adapter_mapping()` 会根据 `LoRAMappingType` 选择目标 wrapper：

```python
if not (self.supports_mm and self.supports_tower_connector_lora):
    target_prefix = ...
elif mapping.type == LoRAMappingType.TOWER and self.mm_mapping.tower_model:
    target_prefix = self.mm_mapping.tower_model[0]
elif mapping.type == LoRAMappingType.CONNECTOR and self.mm_mapping.connector:
    target_prefix = self.mm_mapping.connector[0]
else:
    target_prefix = self.mm_mapping.language_model[0]
```

位置：`model_manager.py:344` 到 `model_manager.py:367`

`GPUModelRunner._execute_mm_encoder()` 会为 encoder 输入单独设置 TOWER / CONNECTOR mapping。

位置：`gpu_model_runner.py:2941` 到 `gpu_model_runner.py:3008`

这说明多模态 LoRA 的并行不仅发生在 language model 的 token batch，还发生在 encoder batch，且 encoder batch 的 shape / token count 由 multimodal item 决定。

---

## 16. 总结

LoRA 和并行机制的关系可以压缩成一句话：

```text
LoRA 权重不是作为一个完整 adapter 广播后直接使用，而是通过与 base parallel layer 匹配的 LoRA wrapper，在每个 rank 上写入本地分片，并用统一的 lora_int_id mapping 在 forward 时选择对应 slot。
```

分并行方式看：

| 并行方式 | LoRA 处理方式 |
|---|---|
| TP ColumnParallel | LoRA B 按输出维切；fully sharded 时 LoRA A 也按 rank 维切，并需要 all_gather shrink buffer |
| TP RowParallel | LoRA A 按输入维切；fully sharded 时 LoRA B 也按输出维切，并配合 all_reduce |
| QKV / Merged | 按 Q/K/V 或 gate/up 等 packed slice 精确切 LoRA B |
| ReplicatedLinear | A/B 都复制，不随 TP 切分 |
| VocabParallelEmbedding | 使用 embedding 专用 LoRA wrapper 和 embedding delta 路径 |
| PP | 当前 PP rank 只包装和加载自己持有的层，跳过 `PPMissingLayer` |
| DP | LoRA 控制面通过 executor/worker 广播保持 adapter id 语义一致，请求只在所在 DP replica 激活本地 mapping |
| EP / MoE | 当前 EP rank 只加载本地 expert 的 LoRA 权重，非本地 expert checkpoint key 可跳过 |

最终，LoRA 并行正确性的关键是：

```text
adapter id 一致；
wrapper 类型匹配 base layer；
权重切片匹配 rank；
batch mapping 匹配 token 顺序。
```
