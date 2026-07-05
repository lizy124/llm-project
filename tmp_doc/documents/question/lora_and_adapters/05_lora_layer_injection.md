# 05. LoRA layer 如何注入 Linear / Embedding / LM head？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\lora_model_runner_mixin.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\worker_manager.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\model_manager.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\lora_model.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\lora_weights.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\layers\base.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\layers\base_linear.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\layers\column_parallel_linear.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\layers\row_parallel_linear.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\layers\replicated_linear.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\layers\vocal_parallel_embedding.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\layers\logits_processor.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\punica_wrapper\punica_base.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\punica_wrapper\punica_gpu.py`

说明：当前 vLLM 代码里，旧文档常说的 `vllm/lora/layers.py`、`vllm/lora/models.py` 已经拆分为 `vllm/lora/layers/*`、`lora_model.py`、`model_manager.py`。本文按当前代码路径梳理。

本问题关注：LoRA adapter 如何挂到模型 layer 上；哪些 Linear / Embedding / LM head / MoE 模块会被替换；adapter 权重如何装入 wrapper 的 GPU slot；forward 时如何按每个 token / prompt 的 LoRA id 选择对应 A/B 权重，并在 base output 上叠加低秩 delta。

---

## 1. 一句话回答

vLLM 的 LoRA 注入不是在每次 forward 时临时查找模块，而是在模型加载阶段先把可支持的 layer 替换成 LoRA-aware wrapper：

```text
load base model
  → LoRAModelRunnerMixin.load_lora_model()
  → WorkerLoRAManager.create_lora_manager()
  → LoRAModelManager.__init__()
  → _create_lora_modules()
  → 遍历 model.named_modules()
  → 按 supported_lora_modules + target_modules 过滤
  → from_layer(...) 选择具体 LoRA wrapper
  → replace_submodule(...) 替换原模块
  → wrapper.create_lora_weights(max_loras, lora_config)
  → wrapper.set_mapping(punica_wrapper)
```

运行时：

```text
请求携带 LoRARequest
  → InputBatch 记录 request_lora_mapping
  → make_lora_inputs() 生成 token_lora_mapping / prompt_lora_mapping
  → set_active_adapters(...)
  → 必要时从 adapter checkpoint 加载 LoRAModel
  → activate_adapter() 把 A/B 权重复制到 wrapper 的某个 GPU slot
  → punica_wrapper.update_metadata(...) 把 token/prompt 映射转换为 kernel metadata
  → wrapper.forward() 先算 base output，再通过 punica kernel 加 LoRA delta
```

核心公式仍然是：

```text
LoRA delta = x @ A @ B * scaling
output = base_output + LoRA delta
```

但在 vLLM 中，真实执行多了几层工程约束：

```text
- 一个 batch 可以混多个 LoRA adapter；
- 每个 token 可能对应不同 adapter；
- TP 下 A/B 需要按 ColumnParallel / RowParallel 的分片方式切片；
- qkv_proj / gate_up_proj 这类 packed layer 需要把多个 adapter-visible 子模块合并成一个 runtime wrapper；
- embedding 和 lm_head 不是普通 Linear 路径；
- MoE 有 2D / 3D / EP / non-gated 等特殊布局。
```

---

## 2. 最小主链路

### 2.1 模型加载时：创建 LoRA manager 并替换 layer

入口在 V1 runner mixin：

```python
def load_lora_model(self, model, vllm_config, device):
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

`WorkerLoRAManager.create_lora_manager()` 会继续调用 `lora/model_manager.py` 中的 `create_lora_manager(...)`，最终构造 `LoRAModelManager`。在 `LoRAModelManager.__init__()` 里，关键初始化顺序是：

```text
get_supported_lora_modules(model)
process_packed_modules_mapping(model)
_init_punica_wrapper(...)
_create_lora_modules()
model.lora_manager = self
```

位置：`model_manager.py:88` 到 `model_manager.py:137`

其中真正替换 layer 的函数是：

```python
def _create_lora_modules(self):
```

位置：`model_manager.py:375` 到 `model_manager.py:502`

### 2.2 请求运行时：设置 active LoRA 和 mapping

`GPUInputBatch.make_lora_inputs()` 根据当前 batch 中每个 request 的 LoRA id，生成两类映射：

```python
req_lora_mapping = self.request_lora_mapping[: self.num_reqs]
prompt_lora_mapping = tuple(req_lora_mapping.repeat(num_sampled_tokens))
token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))
```

位置：`gpu_input_batch.py:976` 到 `gpu_input_batch.py:999`

含义：

```text
token_lora_mapping：
  长度等于本轮 scheduled tokens 数；
  Linear / Embedding 等 token 级计算使用它。

prompt_lora_mapping：
  长度等于本轮 sampled tokens 数；
  logits / sampler 侧使用它。
```

`LoRAModelRunnerMixin.set_active_loras()` 把它们包装成 `LoRAMapping`：

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

### 2.3 adapter 加载和激活

`WorkerLoRAManager.set_active_adapters()` 做两件事：

```python
self._apply_adapters(requests)
self._adapter_manager.set_adapter_mapping(mapping)
```

位置：`worker_manager.py:183` 到 `worker_manager.py:186`

`_apply_adapters()` 会保证本轮需要的 adapter 已经加载，没用的 adapter 被移除：

```text
existing_adapters - requested_ids
  → remove_adapter(...)

requested_ids - existing_adapters
  → add_adapter(...)
  → _load_adapter(...)
  → LoRAModel.from_local_checkpoint(...)
  → LoRAModelManager.add_adapter(...)
  → LoRAModelManager.activate_adapter(...)
```

位置：`worker_manager.py:194` 到 `worker_manager.py:219`

`activate_adapter()` 会把 LoRA 权重复制进每个 wrapper 预分配的 slot：

```python
for module_name, module in self.modules.items():
    module_lora = self._get_lora_layer_weights(lora_model, module_name)
    if not module_lora:
        module.reset_lora(index)
        continue

    module.set_lora(index, module_lora.lora_a, module_lora.lora_b)
```

位置：`model_manager.py:285` 到 `model_manager.py:324`

### 2.4 forward 时：wrapper 使用 punica metadata 选择 adapter

`LoRAModelManager.set_adapter_mapping()` 会更新对应 punica wrapper 的 metadata：

```python
punica_wrapper.update_metadata(
    mapping,
    self.lora_index_to_id,
    self.lora_slots + 1,
    self.vocab_size,
)
```

位置：`model_manager.py:344` 到 `model_manager.py:367`

`PunicaWrapperGPU.update_metadata()` 会把 LoRA mapping 转成 kernel 需要的 token mapping metadata：

```python
self.is_prefill = mapping.is_prefill
self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)
self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)
```

位置：`punica_gpu.py:75` 到 `punica_gpu.py:88`

最终每个 LoRA wrapper 的 forward 都会复用这些 metadata，按 token / prompt 选择当前应该用哪个 LoRA slot。

---

## 3. 哪些模块会被注入 LoRA

### 3.1 vLLM 先计算 supported LoRA module suffix

`get_supported_lora_modules(model)` 会遍历 `model.named_modules()`：

```python
supported_lora_modules = set()

for name, module in model.named_modules():
    embedding_modules = getattr(module, "embedding_modules", None)
    if embedding_modules is not None:
        for name in embedding_modules:
            supported_lora_modules.add(name)

    if isinstance(module, LinearBase):
        supported_lora_modules.add(name.split(".")[-1])

    if isinstance(module, MoERunner):
        supported_lora_modules.add(name.split(".")[-1])
```

位置：`lora/utils.py:208` 到 `lora/utils.py:229`

这说明 vLLM 默认支持三类模块：

```text
1. 模型声明的 embedding_modules
   例如 embed_tokens / lm_head。

2. 所有 LinearBase 子类
   例如 q_proj / k_proj / v_proj / o_proj / gate_up_proj / down_proj 等。

3. MoERunner
   用于 fused MoE 层。
```

### 3.2 再按部署时 `target_modules` 过滤

`_match_target_modules()` 先看模块是否在 vLLM 支持范围内，再看用户的 `LoRAConfig.target_modules` 是否允许：

```python
if not is_supported_lora_module(module_name, self.supported_lora_modules):
    return False
return is_in_target_modules(
    module_name,
    self.lora_config.target_modules,
    self.packed_modules_mapping,
)
```

位置：`model_manager.py:672` 到 `model_manager.py:692`

`is_supported_lora_module()` 用 suffix regex 匹配：

```python
return any(
    re.match(r".*\.{target_module}$", module_name)
    or target_module == module_name
    for target_module in supported_lora_modules
)
```

位置：`lora/utils.py:232` 到 `lora/utils.py:257`

`is_in_target_modules()` 的关键是支持 packed module 映射：

```python
if target_modules is None:
    return True

if module_suffix in target_module_set or module_name in target_module_set:
    return True

packed_children = packed_modules_mapping.get(module_suffix)
if packed_children and any(child in target_module_set for child in packed_children):
    return True

return any(
    module_suffix in children and packed_parent in target_module_set
    for packed_parent, children in packed_modules_mapping.items()
)
```

位置：`lora/utils.py:260` 到 `lora/utils.py:300`

这解决了一个常见问题：

```text
checkpoint / PEFT target_modules 里可能写 q_proj、k_proj、v_proj；
但 vLLM runtime 模型里实际模块可能是 qkv_proj。

checkpoint / PEFT target_modules 里可能写 gate_proj、up_proj；
但 vLLM runtime 模型里实际模块可能是 gate_up_proj。
```

因此 target_modules 可以匹配 packed parent，也可以匹配 adapter-visible child。

### 3.3 packed_modules_mapping 从模型类来

`process_packed_modules_mapping()` 对普通模型返回 `get_packed_modules_mapping(model)`：

```python
return get_packed_modules_mapping(model)
```

位置：`lora/utils.py:360` 到 `lora/utils.py:392`

典型模型会声明：

```text
packed_modules_mapping = {
  "qkv_proj": ["q_proj", "k_proj", "v_proj"],
  "gate_up_proj": ["gate_proj", "up_proj"],
}
```

LoRA manager 会把 runtime packed module 注册到 `self.packed_modules`：

```python
replacements = self.packed_modules_mapping.get(module_name, [])
if len(replacements) <= 1:
    return
prefix = ".".join(parts[:-1])
self.packed_modules[module_full_name] = [
    prefix + "." + r if prefix else r for r in replacements
]
```

位置：`model_manager.py:711` 到 `model_manager.py:722`

后续 adapter 权重加载后，会把这些 child LoRA 合并成 packed layer 的 LoRA 权重。

---

## 4. `_create_lora_modules()` 的注入流程

### 4.1 遍历模块并过滤

核心循环：

```python
for module_name, module in self.model.named_modules(remove_duplicate=False):
    if isinstance(module, PPMissingLayer):
        continue

    if not self._match_target_modules(module_name):
        continue

    punica_wrapper = self._get_punica_wrapper(module_name)
    if punica_wrapper is None:
        continue
```

位置：`model_manager.py:385` 到 `model_manager.py:400`

几个细节：

```text
remove_duplicate=False：
  同一个底层 module 可能通过多个路径出现，后面会处理 alias。

PPMissingLayer：
  pipeline parallel 中当前 rank 不持有的 layer 不替换。

_get_punica_wrapper：
  普通语言模型使用默认 wrapper；
  多模态模型可能按 language / tower / connector 使用不同 wrapper。
```

### 4.2 处理同一 module 的多个 alias

如果同一个底层 module 已经被 wrapper 包过，且不是 `lm_head`，vLLM 会把 alias 属性也指向同一个 wrapper：

```python
existing_wrapper = wrapped_by_id.get(id(module))
if existing_wrapper is not None and "lm_head" not in module_name:
    parent = self.model.get_submodule(_parent_module(module_name))
    setattr(parent, module_name.rpartition(".")[-1], existing_wrapper)
    continue
```

位置：`model_manager.py:415` 到 `model_manager.py:430`

注释里举的场景是 MoE gate 同时挂在 block 和 MoE runner 内。不能为同一个底层 module 注册两份 `self.modules`，否则 `activate_adapter()` 可能先 set 再 reset，导致权重被擦掉。

### 4.3 根据 layer 类型选择 wrapper

注入时调用：

```python
new_module = replace_submodule(
    self.model,
    module_name,
    from_layer(
        module,
        self.lora_slots,
        self.lora_config,
        packed_moduled_lst,
        self.model.config,
    ),
)
```

位置：`model_manager.py:441` 到 `model_manager.py:451`

`replace_submodule()` 非常直接：

```python
parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
target_name = module_name.split(".")[-1]
setattr(parent, target_name, new_module)
return new_module
```

位置：`lora/utils.py:145` 到 `lora/utils.py:152`

`from_layer()` 会依次尝试所有 wrapper：

```python
for lora_cls in _all_lora_classes:
    if lora_cls.can_replace_layer(...):
        instance_layer = lora_cls(layer)
        instance_layer.create_lora_weights(max_loras, lora_config, model_config)
        return instance_layer
return layer
```

位置：`lora/utils.py:106` 到 `lora/utils.py:124`

wrapper 顺序很重要，源码也明确说明 “more specific wrappers must be checked before generic wrappers”：

```text
VocabParallelEmbeddingWithLoRA
ColumnParallelLinearWithLoRA
MergedColumnParallelLinearWithLoRA
QKVParallelLinearWithLoRA
MergedQKVParallelLinearWithLoRA
RowParallelLinearWithLoRA
ReplicatedLinearWithLoRA
LogitsProcessorWithLoRA
ColumnParallelLinearWithShardedLoRA
QKVParallelLinearWithShardedLoRA
MergedColumnParallelLinearWithShardedLoRA
MergedColumnParallelLinearVariableSliceWithLoRA
MergedQKVParallelLinearWithShardedLoRA
RowParallelLinearWithShardedLoRA
FusedMoEWithLoRA
FusedMoE3DWithLoRA
```

位置：`lora/utils.py:76` 到 `lora/utils.py:95`

### 4.4 `lm_head` 还会额外替换 `logits_processor`

如果当前模块名包含 `lm_head`，vLLM 还会找到同级的 `logits_processor`，替换成 `LogitsProcessorWithLoRA`：

```python
if "lm_head" in module_name:
    logits_processor_module_name = "logits_processor"
    ...
    logits_processor_module = self.model.get_submodule(logits_processor_module_name)

    new_module = replace_submodule(
        self.model,
        logits_processor_module_name,
        from_layer_logits_processor(...),
    )
```

位置：`model_manager.py:456` 到 `model_manager.py:479`

这是因为 LM head 的 LoRA delta 最终影响的是 logits。vLLM 不只是替换 `lm_head` embedding/linear 本体，还要让 logits processor 在 `_get_logits()` 中叠加 LoRA 输出。

### 4.5 注册 wrapper 并绑定 punica wrapper

如果替换后的模块是 `BaseLayerWithLoRA`：

```python
self.register_module(module_name, new_module)
self._register_packed_modules(module_name)
new_module.set_mapping(punica_wrapper)
```

位置：`model_manager.py:497` 到 `model_manager.py:501`

`set_mapping()` 只是把 punica wrapper 引用挂到 layer 上：

```python
def set_mapping(self, punica_wrapper):
    self.punica_wrapper = punica_wrapper
```

位置：`layers/base.py:63` 到 `layers/base.py:67`

所以注入阶段最终留下两份关键状态：

```text
LoRAModelManager.modules：
  module_name -> BaseLayerWithLoRA wrapper
  用于 activate_adapter() 时把 A/B 权重写入 wrapper slot。

BaseLayerWithLoRA.punica_wrapper：
  wrapper forward 时读取当前 token/prompt 的 LoRA mapping metadata。
```

---

## 5. LoRA 权重如何从 checkpoint 进入 wrapper

### 5.1 checkpoint 加载为 `LoRAModel`

`WorkerLoRAManager._load_adapter()` 会：

```text
1. 根据 supported_lora_modules + packed_modules_mapping 构造 expected_lora_modules；
2. 读取 adapter_config.json 得到 PEFTHelper；
3. validate_legal(lora_config)；
4. 读取 adapter_model.safetensors / .bin / .pt；
5. LoRAModel.from_local_checkpoint(...)；
6. 生成 LoRAModel(id, rank, loras)。
```

位置：`worker_manager.py:99` 到 `worker_manager.py:162`

`LoRAModel.from_local_checkpoint()` 会先校验 checkpoint 里的模块是否在 expected modules 中：

```python
module_name, _ = parse_fine_tuned_lora_name(lora_module, weights_mapper)
if module_name.rsplit(".", 1)[-1] not in expected_lora_modules:
    unexpected_modules.append(module_name)
```

位置：`lora_model.py:167` 到 `lora_model.py:306`

### 5.2 PEFT 权重名解析

`parse_fine_tuned_lora_name()` 支持两类命名：

```text
...lora_A.weight / ...lora_B.weight
...lora_embedding_A / ...lora_embedding_B
```

位置：`lora/utils.py:155` 到 `lora/utils.py:196`

它返回：

```text
module_name：目标模块名，例如 model.layers.0.self_attn.q_proj
is_lora_a：当前 tensor 是 A 还是 B
```

如果模型有 `hf_to_vllm_mapper`，加载前会先做 name mapping。`WorkerLoRAManager._load_adapter()` 会从模型上读取：

```python
hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
```

位置：`worker_manager.py:124` 到 `worker_manager.py:143`

这对 Qwen2VL 等 HF 名称和 vLLM 模块名前缀不同的模型很重要。

### 5.3 `LoRALayerWeights` 保存单层 A/B

`LoRALayerWeights` 保存：

```python
module_name
rank
lora_alpha
lora_a
lora_b
scaling = lora_alpha / rank
```

位置：`lora_weights.py:13` 到 `lora_weights.py:42`

`optimize()` 会把 scaling 乘进 `lora_b`：

```python
if self.scaling != 1:
    self.lora_b *= self.scaling
    self.scaling = 1
```

位置：`lora_weights.py:36` 到 `lora_weights.py:42`

因此 wrapper forward 中传给 punica 的 scale 通常是 `1.0`，因为 scaling 已经被吸收到 B 矩阵里。

### 5.4 packed layer 权重合并

adapter checkpoint 里通常按 PEFT 子模块保存：

```text
q_proj.lora_A / q_proj.lora_B
k_proj.lora_A / k_proj.lora_B
v_proj.lora_A / v_proj.lora_B
```

但 vLLM runtime layer 可能是一个 `qkv_proj`。`_create_merged_loras_inplace()` 会把 child LoRA 合成 packed LoRA：

```python
for module_name, new_module_names in self.packed_modules.items():
    replacement_loras = []
    for r in new_module_names:
        lora = self._get_lora_layer_weights(lora_model, r)
        replacement_loras.append(lora)

    if module_name.endswith(".experts"):
        lora_model.loras[module_name] = PackedLoRALayerWeights.pack_moe(...)
    else:
        lora_model.loras[module_name] = PackedLoRALayerWeights.pack(replacement_loras)

    for module in packed_module_names:
        lora_model.loras.pop(module, None)
```

位置：`model_manager.py:724` 到 `model_manager.py:774`

`PackedLoRALayerWeights.pack()` 会保存 list 形式的 A/B：

```python
lora_a = [q_A, k_A, v_A]
lora_b = [q_B, k_B, v_B]
```

位置：`lora_weights.py:99` 到 `lora_weights.py:152`

### 5.5 激活 adapter：复制到 GPU slot

LoRA manager 维护：

```python
self.lora_index_to_id: list[int | None] = [None] * self.lora_slots
```

位置：`model_manager.py:103`

`activate_adapter()` 找一个空 slot：

```python
first_free_slot = next((i, lora_id) for i, lora_id in enumerate(...) if lora_id is None)
index, _ = first_free_slot
self.lora_index_to_id[index] = lora_model.id
```

位置：`model_manager.py:285` 到 `model_manager.py:309`

然后对所有已注入模块调用：

```text
module.set_lora(index, module_lora.lora_a, module_lora.lora_b)
```

位置：`model_manager.py:309` 到 `model_manager.py:323`

这意味着：

```text
LoRA checkpoint 权重常驻 CPU cache；
激活后复制到每个 wrapper 的 GPU stacked tensor 某个 slot；
forward 时 kernel 根据 token_lora_mapping 选择 slot，而不是直接拿 checkpoint tensor。
```

---

## 6. Linear LoRA wrapper 如何计算 delta

### 6.1 公共基类：`BaseLinearLayerWithLoRA`

所有 Linear 类 wrapper 继承 `BaseLinearLayerWithLoRA`。

初始化保存 base layer 和并行信息：

```python
self.base_layer = base_layer
self.input_size = self.base_layer.input_size
self.tp_size = self.base_layer.tp_size
self.tp_rank = self.base_layer.tp_rank
self.device = _get_lora_device(self.base_layer)
```

位置：`base_linear.py:69` 到 `base_linear.py:80`

`create_lora_weights()` 预分配 stacked A/B：

```python
self.lora_a_stacked = tuple(
    torch.zeros(max_loras, 1, lora_a_out_size, self.input_size, ...)
    for _ in range(self.n_slices)
)
self.lora_b_stacked = tuple(
    torch.zeros(max_loras, 1, lora_b_out_size, max_lora_rank, ...)
    for _ in range(self.n_slices)
)
```

位置：`base_linear.py:99` 到 `base_linear.py:150`

维度可以理解为：

```text
lora_a_stacked[slice][slot, layer_idx, rank, input_dim]
lora_b_stacked[slice][slot, layer_idx, output_dim, rank]
```

这里 `layer_idx` 固定是 1 维，是为了和 punica kernel 的统一接口兼容。

### 6.2 `set_lora()` 写入 slot

普通 Linear wrapper 的 `set_lora()`：

```python
self.reset_lora(index)
if self.tp_size > 1:
    lora_a = self.slice_lora_a(lora_a)
    lora_b = self.slice_lora_b(lora_b)

self.lora_a_stacked[0][index, 0, : lora_a.shape[0], : lora_a.shape[1]].copy_(lora_a)
self.lora_b_stacked[0][index, 0, : lora_b.shape[0], : lora_b.shape[1]].copy_(lora_b)
```

位置：`base_linear.py:157` 到 `base_linear.py:183`

不同并行层的差别主要体现在 `slice_lora_a()` / `slice_lora_b()`。

### 6.3 forward：base output + LoRA output

同步路径：

```python
output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
return self._apply_lora_to_output(x, output)
```

位置：`base_linear.py:195` 到 `base_linear.py:199`

`_apply_lora_to_output()` 调 punica：

```python
lora_output = self.punica_wrapper.add_lora_linear(
    output,
    x,
    self.lora_a_stacked,
    self.lora_b_stacked,
    1.0,
    self.output_slices,
)
```

位置：`base_linear.py:206` 到 `base_linear.py:229`

如果平台支持 inplace update，punica 直接改 `output`；否则返回新的 `lora_output`。

### 6.4 dual stream 可选路径

如果 `VLLM_LORA_ENABLE_DUAL_STREAM` 开启，`apply()` 会走异步 custom op：

```python
return torch.ops.vllm.lora_linear_async(self.layer_name, output_size, x, bias)
```

位置：`base_linear.py:185` 到 `base_linear.py:193`

异步实现里 base linear 和 LoRA delta 分别在默认 stream / aux stream 上计算，然后：

```python
output.add_(lora_result)
```

位置：`base_linear.py:231` 到 `base_linear.py:295`

这是性能优化，不改变语义。

---

## 7. Column / QKV / Merged Linear 的注入细节

### 7.1 `ColumnParallelLinearWithLoRA`

适用场景：

```text
ColumnParallelLinear
MergedColumnParallelLinear 且 packed_modules_list 长度为 1
```

位置：`column_parallel_linear.py:83` 到 `column_parallel_linear.py:179`

TP 下 ColumnParallel 的 base weight 按输出维切分，所以 LoRA B 也要按输出维切：

```python
shard_size = self.output_size
start_idx = self.tp_rank * shard_size
end_idx = (self.tp_rank + 1) * shard_size
lora_b = lora_b[start_idx:end_idx, :]
```

位置：`column_parallel_linear.py:105` 到 `column_parallel_linear.py:128`

forward 里先计算本 rank 的 output parallel，再按 base layer 配置决定是否 all-gather：

```python
output_parallel = self.apply(input_, bias)
if self.base_layer.gather_output and self.tp_size > 1:
    output = tensor_model_parallel_all_gather(output_parallel)
else:
    output = output_parallel
```

位置：`column_parallel_linear.py:130` 到 `column_parallel_linear.py:156`

### 7.2 `MergedColumnParallelLinearWithLoRA`

适用场景：

```text
gate_proj + up_proj → gate_up_proj
```

或者其他两个 output slice 合并的 ColumnParallel layer。

类注释说明：

```text
ColumnParallelLinear layer that is composed of 2 sublayers packed together.
This means we have 2 LoRAs, each applied to one half of the layer.
```

位置：`column_parallel_linear.py:182` 到 `column_parallel_linear.py:189`

它会按 `base_layer.output_sizes` 创建多个 slice：

```python
self.output_slices = tuple(divide(output_size, self.tp_size) for output_size in self.output_sizes)
self.n_slices = len(self.output_slices)
```

位置：`column_parallel_linear.py:191` 到 `column_parallel_linear.py:203`

`set_lora()` 支持 list 形式的 A/B，并逐 slice 写入：

```python
for i in range(self.n_slices):
    if lora_a_i is not None:
        self.lora_a_stacked[i][index, 0, ...].copy_(lora_a_i)
    if lora_b_i is not None:
        self.lora_b_stacked[i][index, 0, ...].copy_(lora_b_i)
```

位置：`column_parallel_linear.py:300` 到 `column_parallel_linear.py:327`

### 7.3 `QKVParallelLinearWithLoRA`

适用场景：

```text
runtime 是 QKVParallelLinear
packed_modules_list 长度为 1
```

位置：`column_parallel_linear.py:365` 到 `column_parallel_linear.py:428`

这类用于某些模型把 qkv 当成一个 adapter target 的情况。TP 下 Q/K/V 的 shard 逻辑不同，尤其 GQA/MQA 中 KV heads 可能复制。`slice_lora_b()` 会分别切 Q、K、V：

```python
lora_b_q = lora_b[q_shard]
lora_b_k = lora_b[k_offset + kv_shard]
lora_b_v = lora_b[v_offset + kv_shard]
lora_b = torch.cat([lora_b_q, lora_b_k, lora_b_v], dim=0)
```

位置：`column_parallel_linear.py:393` 到 `column_parallel_linear.py:414`

### 7.4 `MergedQKVParallelLinearWithLoRA`

适用场景：

```text
q_proj + k_proj + v_proj → qkv_proj
packed_modules_list 长度为 3
```

位置：`column_parallel_linear.py:431` 到 `column_parallel_linear.py:489`

它有 3 个 slice：

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

位置：`column_parallel_linear.py:442` 到 `column_parallel_linear.py:463`

所以 q/k/v 三份 LoRA 权重会分别写入三个 stacked slice。

### 7.5 fully sharded LoRA

源码中带 `WithShardedLoRA` 的类对应 `lora_config.fully_sharded_loras=True`。

判断由 decorator 控制：

```python
_fully_sharded_can_replace(...)
_not_fully_sharded_can_replace(...)
```

位置：`layers/utils.py:73` 到 `layers/utils.py:98`

区别：

```text
普通 ColumnParallel：
  LoRA A 通常不切，LoRA B 按输出切。

fully sharded ColumnParallel：
  LoRA A 也沿 rank 维切，计算后需要 all-gather shrink buffer。

普通 RowParallel：
  LoRA A 按输入切，LoRA B 不切。

fully sharded RowParallel：
  LoRA B 也按输出切，结合 all-reduce 语义。
```

Column fully sharded 的 `slice_lora_a()` 见 `column_parallel_linear.py:497` 到 `column_parallel_linear.py:516`。

Row fully sharded 的 `slice_lora_b()` 和 `apply()` 见 `row_parallel_linear.py:101` 到 `row_parallel_linear.py:177`。

---

## 8. Row / Replicated Linear 的注入细节

### 8.1 `RowParallelLinearWithLoRA`

RowParallel 的 base weight 按输入维切分，所以 LoRA A 需要按输入维切：

```python
shard_size = self.input_size
start_idx = self.tp_rank * shard_size
end_idx = (self.tp_rank + 1) * shard_size
lora_a = lora_a[:, start_idx:end_idx]
```

位置：`row_parallel_linear.py:22` 到 `row_parallel_linear.py:40`

forward 逻辑保持 RowParallel 原语义：

```python
if self.base_layer.input_is_parallel:
    input_parallel = input_
else:
    split_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
    input_parallel = split_input[self.tp_rank].contiguous()

output_parallel = self.apply(input_parallel, bias_)
if self.base_layer.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
```

位置：`row_parallel_linear.py:42` 到 `row_parallel_linear.py:82`

### 8.2 `ReplicatedLinearWithLoRA`

ReplicatedLinear 每张卡都有完整权重，所以不需要 TP 切片：

```python
def slice_lora_a(...): return lora_a
def slice_lora_b(...): return lora_b
```

位置：`replicated_linear.py:16` 到 `replicated_linear.py:78`

它的 `apply()` 走 `_apply_base_forward()`：

```python
return self._apply_base_forward(x)
```

位置：`replicated_linear.py:49` 到 `replicated_linear.py:53`

原因是某些 `ReplicatedLinear` 子类会重写 forward，调用自定义 kernel 或调整 dtype。这里先尊重 base layer 的真实 forward 输出，再叠加 LoRA。

---

## 9. Embedding LoRA 如何注入

### 9.1 wrapper 类型

Embedding 对应：

```python
class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
```

位置：`vocal_parallel_embedding.py:17` 到 `vocal_parallel_embedding.py:140`

注意文件名是 `vocal_parallel_embedding.py`，类名是 `VocabParallelEmbeddingWithLoRA`。

它能替换：

```python
return type(source_layer) is maybe_get_oot_by_class(VocabParallelEmbedding)
```

位置：`vocal_parallel_embedding.py:128` 到 `vocal_parallel_embedding.py:136`

### 9.2 预分配权重

Embedding LoRA 的 A 不是 `[rank, input_dim]` 形式，而是按 vocab id 查表：

```python
self.lora_a_stacked = torch.zeros(
    max_loras,
    self.base_layer.org_vocab_size,
    max_lora_rank,
)
self.lora_b_stacked = torch.zeros(
    max_loras,
    1,
    embedding_dim,
    max_lora_rank,
)
```

位置：`vocal_parallel_embedding.py:24` 到 `vocal_parallel_embedding.py:72`

`set_lora()` 里会把 PEFT 的 `lora_embedding_A` 转置后写入：

```python
self.lora_a_stacked[index, : lora_a.shape[1], : lora_a.shape[0]].copy_(lora_a.T)
self.lora_b_stacked[index, 0, : lora_b.shape[0], : lora_b.shape[1]].copy_(lora_b)
```

位置：`vocal_parallel_embedding.py:77` 到 `vocal_parallel_embedding.py:94`

### 9.3 forward：先查 LoRA A embedding，再 expand

Embedding forward：

```python
indices_1 = self.punica_wrapper._embeddings_indices[1][:num_tokens]

full_lora_a_embeddings = F.embedding(
    x + indices_1,
    self.lora_a_stacked_2d,
)
full_output = self.base_layer.forward(x)

self.punica_wrapper.add_lora_embedding(
    full_output,
    full_lora_a_embeddings,
    self.lora_b_stacked,
    add_input=True,
)
```

位置：`vocal_parallel_embedding.py:96` 到 `vocal_parallel_embedding.py:126`

可以理解为：

```text
base embedding：
  y = E[token_id]

LoRA embedding：
  a = A_lora[token_id]
  delta = a @ B_lora
  y = y + delta
```

`_embeddings_indices` 来自 punica 的 mapping 转换，用于把不同 LoRA slot 的 embedding A 展平成一个大表后仍能按 token 取到正确 adapter 的 A。

---

## 10. LM head / logits_processor LoRA 如何注入

### 10.1 为什么要替换 logits_processor

LM head 最终用于计算 logits。vLLM 中 logits 处理不只是一个裸 linear，后面还涉及 TP gather、vocab padding、sharded-to-full reindex 等逻辑。因此 `lm_head` 命中 LoRA 后，manager 会额外替换 `logits_processor`：

```python
from_layer_logits_processor(
    logits_processor_module,
    module,
    self.lora_slots,
    self.lora_config,
    self.model.config,
)
```

位置：`model_manager.py:456` 到 `model_manager.py:479`

`from_layer_logits_processor()` 构造：

```python
LogitsProcessorWithLoRA(
    layer,
    lm_head.embedding_dim,
    lm_head.weight.dtype,
    lm_head.weight.device,
    lm_head.get_sharded_to_full_mapping(),
)
```

位置：`lora/utils.py:127` 到 `lora/utils.py:142`

### 10.2 `LogitsProcessorWithLoRA` 的权重形状

```python
self.lora_a_stacked = torch.zeros(max_loras, 1, rank, hidden_size)
self.lora_b_stacked = torch.zeros(max_loras, 1, vocab_size, rank)
```

位置：`logits_processor.py:84` 到 `logits_processor.py:120`

这里的输出维是 vocab size，而不是普通 hidden size。

### 10.3 `_get_logits()` 中叠加 LoRA

`LogitsProcessorWithLoRA._get_logits()` 先用实际 lm_head 算 base logits：

```python
logits = actual_lm_head.quant_method.apply(actual_lm_head, hidden_states)
logits = self.base_layer._gather_logits(logits)
```

位置：`logits_processor.py:141` 到 `logits_processor.py:160`

如果有 sharded vocab mapping，会先 reindex：

```python
logits = logits[:, self.sharded_to_full_mapping_gpu]
```

位置：`logits_processor.py:162` 到 `logits_processor.py:180`

然后加 LoRA logits：

```python
self.punica_wrapper.add_lora_logits(
    logits,
    hidden_states,
    self.lora_a_stacked,
    self.lora_b_stacked,
    1.0,
)
```

位置：`logits_processor.py:181` 到 `logits_processor.py:190`

和 Linear 不同，logits 使用的是 `prompt_mapping_meta` / sampler 侧 mapping，而不是 token 侧 mapping。

---

## 11. Punica wrapper 的作用

### 11.1 它不是权重容器，而是 metadata + kernel 调用器

`PunicaWrapperBase` 初始化时分配多种 mapping buffer：

```python
_token_lora_indices
_sampler_indices
_sampler_indices_padded
_embeddings_indices
_seq_start_locs
_seq_lengths
_lora_indices_per_batch
```

位置：`punica_base.py:124` 到 `punica_base.py:167`

它维护的是：

```text
当前 batch 中每个 token 用哪个 LoRA slot；
当前 logits / sampling 位置用哪个 LoRA slot；
embedding 查表时 token id 如何偏移到对应 LoRA A；
prefill kernel 需要的 sequence grouping metadata。
```

真正的 LoRA A/B 权重在各个 layer wrapper 的 `lora_a_stacked` / `lora_b_stacked` 中。

### 11.2 `update_metadata()` 把 LoRA id 转成 slot index

`LoRAMapping` 中保存的是 request 级 LoRA id，例如 1、2、3。`lora_index_to_id` 保存 GPU slot 到 LoRA id 的映射：

```text
slot 0 -> LoRA id 17
slot 1 -> LoRA id 25
slot 2 -> None
```

`convert_mapping(...)` 会把 token/prompt 的 LoRA id 转成 kernel 能用的 slot index，并生成 embedding offset 等辅助索引。

调用位置：`punica_base.py:168` 到 `punica_base.py:203`。

GPU wrapper 随后准备 Triton kernel metadata：

```python
self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)
```

位置：`punica_gpu.py:75` 到 `punica_gpu.py:88`

### 11.3 Linear 的 shrink / expand

GPU 上的 `add_lora_linear()` 分两步：

```python
buffer = torch.empty((len(output_slices), x.size(0), rank), dtype=torch.float32)
self.add_shrink(buffer, x, lora_a_stacked, scale)
self.add_expand(y, buffer, lora_b_stacked, output_slices, add_inputs=True)
```

位置：`punica_gpu.py:203` 到 `punica_gpu.py:265`

语义是：

```text
shrink：
  buffer[slice, token, rank] = x[token] @ A[slot]

expand：
  y[token, output_slice] += buffer[slice, token] @ B[slot]
```

`add_shrink()` 使用 `token_mapping_meta`：

位置：`punica_gpu.py:90` 到 `punica_gpu.py:122`

`add_expand()` 也使用 `token_mapping_meta`：

位置：`punica_gpu.py:123` 到 `punica_gpu.py:170`

### 11.4 Embedding 和 logits 的专用入口

Embedding：

```python
add_lora_embedding(y, x, lora_b_stacked, add_inputs=True)
```

位置：`punica_gpu.py:171` 到 `punica_gpu.py:201`

Logits：

```python
add_lora_logits(y, x, lora_a_stacked, lora_b_stacked, scale)
```

位置：`punica_gpu.py:266` 到 `punica_gpu.py:325`

Logits 使用 `prompt_mapping_meta`：

```python
self.prompt_mapping_meta.meta_args(...)
```

位置：`punica_gpu.py:306` 到 `punica_gpu.py:323`

---

## 12. 多模态模型中的 tower / connector LoRA

普通语言模型只有一个默认 punica wrapper：

```python
self.punica_wrapper_mapping[DEFAULT_LANGUAGE_WRAPPER_KEY] = llm_punica_wrapper
```

位置：`model_manager.py:139` 到 `model_manager.py:163`

多模态模型如果实现 `get_mm_mapping()`，manager 会按模块前缀创建多个 wrapper：

```text
language_model prefix -> LLM wrapper
tower_model prefix    -> tower wrapper
connector prefix      -> connector wrapper
```

位置：`model_manager.py:164` 到 `model_manager.py:269`

是否启用 tower / connector LoRA 还取决于：

```text
model supports_multimodal(model)
model has get_mm_mapping()
model has get_num_mm_encoder_tokens()
lora_config.enable_tower_connector_lora=True
不是 language_model_only 模式
```

位置：`model_manager.py:142` 到 `model_manager.py:151`，以及 `model_manager.py:188` 到 `model_manager.py:225`

模块名到 wrapper 的选择使用最长前缀匹配：

```python
for prefix in sorted(self.punica_wrapper_mapping.keys(), key=len, reverse=True):
    if module_name.startswith(prefix):
        return self.punica_wrapper_mapping[prefix]
```

位置：`model_manager.py:694` 到 `model_manager.py:709`

多模态 encoder 执行时，`GPUModelRunner._execute_mm_encoder()` 会为 tower / connector 设置不同类型的 `LoRAMapping`：

```text
LoRAMappingType.TOWER
LoRAMappingType.CONNECTOR
```

对应逻辑在 `gpu_model_runner.py:2941` 到 `gpu_model_runner.py:3008`。

这和普通 decoder forward 的 `LoRAMappingType.LANGUAGE` 区分开来，避免 tower token budget 和 language token budget 混用。

---

## 13. MoE LoRA 注入的特殊点

MoE 支持由 `is_moe_model(model)` 判断：

```python
if any(isinstance(module, MoERunner) for module in model.modules()):
    logger.info_once("MoE model detected. Using fused MoE LoRA implementation.")
    return True
```

位置：`lora/utils.py:98` 到 `lora/utils.py:103`

`process_packed_modules_mapping()` 对 MoE 会要求模型实现 expert mapping：

```python
if moe_packed_mapping := get_moe_expert_mapping(model):
    packed_modules_mapping = get_packed_modules_mapping(model)
    packed_modules_mapping["experts"] = [...]
else:
    raise AttributeError("To support LoRA for MoE model, 'get_expert_mapping' must be implemented")
```

位置：`lora/utils.py:360` 到 `lora/utils.py:392`

`_create_lora_modules()` 遇到 `MoERunner` 时会把 `packed_moduled_lst` 特殊设置为：

```python
packed_moduled_lst = ["w13"] if self._is_3d_moe_model else ["w1", "w3"]
```

位置：`model_manager.py:432` 到 `model_manager.py:440`

这只是用来决定实例化 `FusedMoE3DWithLoRA` 还是 `FusedMoEWithLoRA`。

MoE 权重 packing 使用：

```python
PackedLoRALayerWeights.pack_moe(...)
```

位置：`lora_weights.py:154` 到 `lora_weights.py:228`

如果是 expert parallel，加载时还会构造 `MoEEPLoadSpec`，用于跳过非本 rank 的 expert 权重：

位置：`lora_model.py:25` 到 `lora_model.py:58`，以及 `worker_manager.py:132` 到 `worker_manager.py:144`。

MoE 细节很多，但从 layer injection 角度看，关键仍然是：

```text
MoERunner 被替换成 FusedMoEWithLoRA / FusedMoE3DWithLoRA；
expert 子模块 checkpoint 权重会 pack 成 experts 这个 runtime module 的 LoRA 权重；
forward 时 fused MoE LoRA kernel 根据 expert id 和 token_lora_mapping 共同选择权重。
```

---

## 14. 一张表总结各类 wrapper

| 原始层 | LoRA wrapper | 关键点 |
|---|---|---|
| `ColumnParallelLinear` | `ColumnParallelLinearWithLoRA` | TP 下切 `lora_b`，必要时 all-gather output |
| `MergedColumnParallelLinear` | `MergedColumnParallelLinearWithLoRA` | `gate_proj/up_proj` 等 packed slice 分别有 LoRA |
| `QKVParallelLinear` | `QKVParallelLinearWithLoRA` | 单个 qkv LoRA，TP 下按 Q/K/V 各自 shard 切 B |
| `QKVParallelLinear` | `MergedQKVParallelLinearWithLoRA` | `q_proj/k_proj/v_proj` 三个 LoRA slice 合到一个 qkv wrapper |
| `RowParallelLinear` | `RowParallelLinearWithLoRA` | TP 下切 `lora_a`，输出按 RowParallel 语义 all-reduce |
| `ReplicatedLinear` | `ReplicatedLinearWithLoRA` | 不切 A/B，尊重 base layer 自定义 forward |
| `VocabParallelEmbedding` | `VocabParallelEmbeddingWithLoRA` | 先按 token id 查 LoRA A，再 expand 到 embedding dim |
| `LogitsProcessor` | `LogitsProcessorWithLoRA` | 通过 lm_head 命中后额外替换，在 logits 上加 delta |
| `MoERunner` | `FusedMoEWithLoRA` / `FusedMoE3DWithLoRA` | expert 权重 pack，kernel 同时考虑 expert id 和 LoRA id |

---

## 15. 最容易出错的点

```text
1. target_modules 写的是 q_proj/k_proj/v_proj，但 runtime 模块是 qkv_proj。
   需要依赖 packed_modules_mapping 正确映射。

2. LoRA checkpoint 的模块名前缀和 vLLM 模型模块名不同。
   需要模型提供 hf_to_vllm_mapper。

3. lm_head 不只是替换 embedding/linear。
   还要替换 logits_processor，否则 logits 阶段不会叠加 LoRA。

4. TP 下不能直接把完整 A/B 复制进 wrapper。
   Column / Row / QKV / fully sharded 的切片规则不同。

5. 一个底层 module 可能有多个 module_name alias。
   不能注册两份 wrapper，否则 activate_adapter() 可能互相 reset。

6. packed layer 中某些 child 没有 LoRA 是允许的。
   PackedLoRALayerWeights 用 None 表示该 slice 没有 adapter 权重。

7. request LoRA id 不是 GPU slot index。
   punica 的 convert_mapping 会结合 lora_index_to_id 把它转换成 slot。

8. Embedding LoRA 使用 lora_embedding_A/B 命名和查表语义。
   它不是普通 `x @ A @ B` 的 Linear 输入。

9. 多模态 tower / connector LoRA 使用单独 token budget 和 LoRAMappingType。
   不能和 language model token mapping 混用。

10. MoE adapter 有 2D / 3D / EP / non-gated 多种布局。
    注入时 wrapper 类型和权重 packing 必须匹配 checkpoint 格式。
```

---

## 16. 总结

LoRA layer 注入可以分成三层理解：

```text
结构层：
  LoRAModelManager._create_lora_modules()
  找到可支持模块，用 from_layer() 生成 LoRA wrapper，再 replace_submodule() 替换。

权重层：
  LoRAModel.from_local_checkpoint()
  把 PEFT checkpoint 解析成 LoRALayerWeights；
  _create_merged_loras_inplace() 合并 packed layer；
  activate_adapter() 把权重复制进 wrapper 的 GPU slot。

执行层：
  InputBatch.make_lora_inputs()
  生成 token_lora_mapping / prompt_lora_mapping；
  punica_wrapper.update_metadata()
  转成 kernel metadata；
  wrapper.forward()
  在 base output 上叠加 LoRA delta。
```

最终可以压缩成一句：

```text
vLLM 的 LoRA 注入是在模型结构上预先替换 layer，在权重上把多个 adapter 放入固定 GPU slot，在执行上用 punica metadata 为 batch 中每个 token 动态选择 LoRA delta。
```
