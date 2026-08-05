# 01. LoRA / adapter 在 vLLM 中负责什么？

源码位置：

- `vllm/vllm/config/lora.py`
- `vllm/vllm/lora/request.py`
- `vllm/vllm/lora/worker_manager.py`
- `vllm/vllm/lora/model_manager.py`
- `vllm/vllm/lora/lora_model.py`
- `vllm/vllm/lora/lora_weights.py`
- `vllm/vllm/lora/layers/`
- `vllm/vllm/lora/punica_wrapper/`
- `vllm/vllm/v1/worker/lora_model_runner_mixin.py`
- `vllm/vllm/v1/worker/gpu_input_batch.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/gpu_worker.py

本问题关注：LoRA 在 vLLM 中的职责、边界，以及它和 base model、adapter 管理、batch 执行、KV cache、量化、sampling 的关系。

---

## 1. 一句话回答

vLLM 中的 LoRA 是一套“请求级可切换的低秩增量权重执行系统”。

它让多个请求共享同一个常驻 base model，同时每个请求可以选择不同 adapter；执行时，vLLM 根据当前 batch 的 LoRA 映射，把对应 adapter 的低秩 delta 叠加到被 LoRA 包装的 layer 输出上。

主链路可以压缩成：

```text
base model 常驻加载
  → --enable-lora 创建 LoRAConfig
  → GPUModelRunner.load_model() 包装 LoRA manager 和 LoRA layer
  → 用户请求携带 LoRARequest
  → Request / InputBatch 记录每个 request 的 lora_id
  → 每轮执行前 set_active_loras()
  → WorkerLoRAManager 加载 / 缓存 / 激活 adapter
  → LoRAMapping 告诉 kernel 每个 token 用哪个 LoRA
  → LoRA-wrapped layer 执行 base output + LoRA delta
  → logits / sampling / output
```

最小心智模型：

```text
base model：
  常驻、共享、一次加载。

LoRA adapter：
  可动态加载、缓存、卸载、pin；
  请求级选择；
  batch 内可混用；
  forward 时作为低秩增量叠加。

LoRARequest：
  请求想用哪个 adapter 的选择信息，不等于 adapter 已经加载。

LoRA manager：
  负责加载、缓存、激活 adapter，并把 LoRA 权重放进 layer 的 slot。
```

---

## 2. LoRA 解决什么问题

### 2.1 一个 base model 服务多个微调任务

没有 LoRA 时，如果每个任务都需要一份完整微调模型，服务端通常要：

```text
任务 A → 加载完整模型 A
任务 B → 加载完整模型 B
任务 C → 加载完整模型 C
```

这会放大显存、加载时间和部署复杂度。

LoRA 的目标是：

```text
共享一份 base model；
每个任务只加载小得多的 adapter delta；
请求级决定使用哪个 adapter。
```

在 vLLM 里，这对应：

```text
base model weights：
  由正常 model loader 加载。

LoRA weights：
  由 WorkerLoRAManager / LoRAModelManager 加载和管理。
```

### 2.2 支持请求级 adapter 选择

用户请求携带 `LoRARequest`。

位置：`request.py:8`

字段：

```python
lora_name: str
lora_int_id: int
lora_path: str = ""
base_model_name: str | None = None
tensorizer_config_dict: dict | None = None
load_inplace: bool = False
is_3d_lora_weight: bool = False
```

位置：`request.py:25` 到 `request.py:37`

含义：

```text
lora_name：
  adapter 的名字。LoRARequest 的 eq/hash 也是按 name 比较。

lora_int_id：
  adapter 的整数 ID，用于 runtime mapping 和 LoRA slot。
  必须 > 0；0 通常表示无 LoRA。

lora_path：
  adapter checkpoint 路径。

base_model_name：
  可用于记录 adapter 对应的 base model。

load_inplace：
  即使相同 lora_int_id 已存在，也强制重新加载并原地替换。

is_3d_lora_weight：
  MoE LoRA 权重格式标记，在 mixed MoE LoRA 格式下使用。
```

`LoRARequest` 只是“选择 adapter 的请求信息”。它不会自己修改模型，也不会自己加载权重。

### 2.3 支持 batch 内混合多个 LoRA

`InputBatch` 保存每个 request 的 LoRA ID：

```python
self.request_lora_mapping = np.zeros((self.max_num_reqs,), dtype=np.int64)
self.lora_id_to_request_ids: dict[int, set[str]] = {}
self.lora_id_to_lora_request: dict[int, LoRARequest] = {}
```

位置：`gpu_input_batch.py:247` 到 `gpu_input_batch.py:250`

新增 request 时：

```python
if request.lora_request:
    lora_id = request.lora_request.lora_int_id
    self.request_lora_mapping[req_index] = lora_id
    self.lora_id_to_request_ids[lora_id].add(request.req_id)
    self.lora_id_to_lora_request[lora_id] = request.lora_request
else:
    self.request_lora_mapping[req_index] = 0
```

位置：`gpu_input_batch.py:471` 到 `gpu_input_batch.py:482`

执行前会展开成 token 级 mapping：

```python
prompt_lora_mapping = tuple(req_lora_mapping.repeat(num_sampled_tokens))
token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))
active_lora_requests = set(self.lora_id_to_lora_request.values())
```

位置：`gpu_input_batch.py:979` 到 `gpu_input_batch.py:1002`

这就是 batch mixed LoRA 的基础：

```text
同一个 batch 里：
  request 0 用 LoRA 1；
  request 1 无 LoRA；
  request 2 用 LoRA 3；

vLLM 会把 request 级 LoRA 选择展开到 token 级 mapping，
LoRA kernel 根据 mapping 对不同 token 应用不同 adapter。
```

### 2.4 支持运行时加载、卸载、pin

Worker 侧控制接口：

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

位置：`gpu_worker.py:1152` 到 `gpu_worker.py:1162`

ModelRunner mixin 再转给 `lora_manager`：

```python
def add_lora(self, lora_request):
    return self.lora_manager.add_adapter(lora_request)

def remove_lora(self, lora_id):
    return self.lora_manager.remove_adapter(lora_id)

def pin_lora(self, lora_id):
    return self.lora_manager.pin_adapter(lora_id)

def list_loras(self):
    return self.lora_manager.list_adapters()
```

位置：`lora_model_runner_mixin.py:274` 到 `lora_model_runner_mixin.py:288`

所以 LoRA 不只是模型结构上的低秩层，它在 vLLM 里还包含一套 runtime lifecycle 管理。

---

## 3. LoRA 在 vLLM 中处于哪一层

可以把 LoRA 拆成六层。

### 3.1 配置层：LoRAConfig

位置：`config/lora.py:31`

关键字段：

```python
max_lora_rank: MaxLoRARanks = 16
max_loras: int = Field(default=1, ge=1)
fully_sharded_loras: bool = False
max_cpu_loras: int | None = None
lora_dtype: torch.dtype | LoRADType = "auto"
target_modules: list[str] | None = None
default_mm_loras: dict[str, str] | None = None
enable_tower_connector_lora: bool = False
specialize_active_lora: bool = False
enable_mixed_moe_lora_format: bool = False
```

位置：`config/lora.py:34` 到 `config/lora.py:79`

职责：

```text
定义 LoRA 能力是否存在以及资源上限；
定义单 batch 最多多少个 LoRA；
定义最大 rank；
定义 CPU cache 容量；
定义 LoRA dtype；
定义哪些 target modules 允许注入 LoRA；
定义多模态 tower/connector LoRA 和 MoE LoRA 特性。
```

注意 `LoRAConfig` 只有在启用 LoRA 时才进入 `VllmConfig`。CLI 层大致是：

```text
--enable-lora 为 True
  → 构造 LoRAConfig
否则
  → lora_config = None
```

对应 `arg_utils.py:2084` 到 `arg_utils.py:2100`。

### 3.2 请求层：LoRARequest

`LoRARequest` 表达单个请求想使用哪个 adapter。

它随请求进入 Engine / Scheduler / Worker，但职责只到“声明选择”。

它不负责：

```text
- 校验 adapter checkpoint 是否真实存在；
- 加载 adapter 权重；
- 决定 adapter 放入哪个 GPU slot；
- 执行 LoRA delta；
- 修改 sampling 参数。
```

这些事情分别由 worker manager、model manager、LoRA layer 和 sampler 负责。

### 3.3 Worker 管理层：WorkerLoRAManager

位置：`worker_manager.py:26`

`WorkerLoRAManager` 是 worker 侧 adapter 管理入口。

它负责：

```text
- 创建 LoRAModelManager；
- 从 lora_path 加载 adapter checkpoint；
- 用 PEFTHelper 校验 LoRA 配置合法性；
- 把 checkpoint 转成 LoRAModel；
- add / remove / pin / list adapter；
- 每轮执行前确保当前 batch 需要的 adapter 已加载并激活。
```

加载 adapter 的核心流程：

```text
_load_adapter(lora_request)
  → get_adapter_absolute_path(lora_request.lora_path)
  → PEFTHelper.from_local_dir(...)
  → peft_helper.validate_legal(lora_config)
  → LoRAModel.from_local_checkpoint(...)
  → 返回 LoRAModel
```

位置：`worker_manager.py:99` 到 `worker_manager.py:162`

### 3.4 模型管理层：LoRAModelManager

位置：`model_manager.py:64`

`LoRAModelManager` 直接持有被 LoRA 化的模型，并管理 adapter slot。

核心字段：

```text
_registered_adapters：
  已加载到 manager 的 LoRAModel。

_active_adapters：
  当前已经放入 GPU slot、可用于 forward 的 adapters。

lora_index_to_id：
  GPU slot index → lora_int_id。

modules：
  module name → BaseLayerWithLoRA wrapper。

punica_wrapper_mapping：
  不同模型部分对应的 LoRA kernel wrapper。
```

位置：`model_manager.py:88` 到 `model_manager.py:137`

它做两类核心工作。

第一类：把 base model 中支持 LoRA 的 module 替换成 LoRA wrapper。

```python
new_module = replace_submodule(
    self.model,
    module_name,
    from_layer(...),
)
```

位置：`model_manager.py:375` 到 `model_manager.py:502`

第二类：把某个 LoRA adapter 的权重放进每个 LoRA layer 的 slot。

```python
module.set_lora(index, module_lora.lora_a, module_lora.lora_b)
```

位置：`model_manager.py:285` 到 `model_manager.py:324`

### 3.5 Layer 层：BaseLayerWithLoRA / LinearWithLoRA

位置：`lora/layers/base.py:16`

`BaseLayerWithLoRA` 定义 LoRA wrapper 必须支持的操作：

```text
create_lora_weights(max_loras, lora_config)
reset_lora(index)
set_lora(index, lora_a, lora_b)
set_mapping(punica_wrapper)
can_replace_layer(...)
```

位置：`lora/layers/base.py:16` 到 `lora/layers/base.py:79`

以 Linear 为例，LoRA forward 的核心是：

```python
output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
return self._apply_lora_to_output(x, output)
```

位置：`base_linear.py:195` 到 `base_linear.py:199`

然后：

```python
lora_output = self.punica_wrapper.add_lora_linear(
    output, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, self.output_slices
)
```

位置：`base_linear.py:206` 到 `base_linear.py:220`

这说明 LoRA layer 的语义是：

```text
先算 base layer output；
再按当前 token_lora_mapping 计算 LoRA delta；
把 delta 加到 output。
```

不是替换 base weight，也不是重新加载一份完整 layer。

### 3.6 Kernel / mapping 层：LoRAMapping + punica wrapper

`LoRAMapping` 在 ModelRunner 中构造：

```python
lora_mapping = LoRAMapping(
    token_lora_mapping,
    prompt_lora_mapping,
    is_prefill=True,
    type=mapping_type,
)
```

位置：`lora_model_runner_mixin.py:48` 到 `lora_model_runner_mixin.py:67`

随后：

```python
self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
```

位置：`lora_model_runner_mixin.py:67`

`LoRAModelManager._set_adapter_mapping()` 会把 mapping 写进对应 punica wrapper：

```python
punica_wrapper.update_metadata(
    mapping,
    self.lora_index_to_id,
    self.lora_slots + 1,
    self.vocab_size,
)
```

位置：`model_manager.py:344` 到 `model_manager.py:367`

这层解决的问题是：

```text
同一个 batch 的不同 token 应该使用哪个 LoRA slot。
```

---

## 4. LoRA 和 base model 的关系

### 4.1 base model 先正常加载

`GPUModelRunner.load_model()` 中先用普通 model loader 加载 base model：

```python
self.model = model_loader.load_model(
    vllm_config=self.vllm_config,
    model_config=self.model_config,
)
```

位置：`gpu_model_runner.py:5231` 到 `gpu_model_runner.py:5254`

如果启用了 LoRA，再包装：

```python
if self.lora_config:
    self.model = self.load_lora_model(
        self.model, self.vllm_config, self.device
    )
```

位置：`gpu_model_runner.py:5167` 到 `gpu_model_runner.py:5170`

### 4.2 LoRA 不是完整模型热切换

LoRA 不会把 base model 换掉。

它做的是：

```text
base output = base_layer(x)
lora delta = B(A(x)) * scaling
final output = base output + lora delta
```

在当前代码里，Linear wrapper 的执行方式就是：

```text
base_layer.quant_method.apply(...)
  → punica_wrapper.add_lora_linear(...)
```

位置：`base_linear.py:195` 到 `base_linear.py:220`

所以：

```text
base model weights：
  始终是主要模型参数。

LoRA adapter weights：
  是请求级增量参数，挂在支持 LoRA 的 module 上。
```

### 4.3 LoRA 需要模型声明支持

`load_lora_model()` 会检查：

```python
if not supports_lora(model):
    raise ValueError(f"{model.__class__.__name__} does not support LoRA yet.")
```

位置：`lora_model_runner_mixin.py:31` 到 `lora_model_runner_mixin.py:46`

`create_lora_manager()` 也会检查：

```python
if not isinstance(model, SupportsLoRA):
    raise ValueError(f"Model {type(model)} is not supported for LoRA.")
```

位置：`model_manager.py:1250` 到 `model_manager.py:1274`

这说明 LoRA 不是任意 torch module 都自动可用；模型需要暴露 vLLM 认可的 LoRA 支持接口和可替换模块。

---

## 5. LoRA 和 adapter 管理的关系

在 vLLM 当前代码里，LoRA 是 adapter 的具体实现之一，但实际源码主路径集中在 `vllm/lora/`。

可以这样区分：

```text
adapter：
  泛指挂在 base model 上的增量能力。

LoRA adapter：
  使用低秩 A/B 权重实现的 adapter。

LoRARequest：
  请求级 adapter 选择信息。

LoRAModel：
  已加载的 adapter 权重集合。

LoRAModelManager：
  负责把 LoRAModel 注册、激活到模型 slot。

WorkerLoRAManager：
  负责 worker 侧加载 checkpoint 和生命周期控制。
```

本目录名叫 `lora_and_adapters`，但当前 vLLM 代码中没有 `code/vllm/vllm/adapter_commons/` 这个实际目录；梳理时应以 `code/vllm/vllm/lora/` 为准。

---

## 6. LoRA 的执行时主链路

### 6.1 初始化阶段

```text
EngineArgs / CLI
  → enable_lora=True
  → LoRAConfig
  → VllmConfig.lora_config
  → GPUModelRunner.load_model()
  → load base model
  → load_lora_model()
  → LRUCacheWorkerLoRAManager
  → LoRAModelManager
  → replace supported modules with LoRA wrappers
```

关键源码：

```text
config/lora.py:31
lora_model_runner_mixin.py:31
model_manager.py:64
model_manager.py:375
```

### 6.2 请求进入阶段

```text
用户请求
  → LoRARequest
  → EngineCoreRequest.lora_request
  → Request.lora_request
  → NewRequestData.lora_request
  → GPUModelRunner.requests[req_id].lora_request
  → InputBatch.request_lora_mapping
```

新请求进入 ModelRunner 时会保存：

```python
lora_request=new_req_data.lora_request
```

位置：`gpu_model_runner.py:1226` 到 `gpu_model_runner.py:1237`

`InputBatch.add_request()` 再记录 `request_lora_mapping`。

位置：`gpu_input_batch.py:471` 到 `gpu_input_batch.py:482`

### 6.3 每轮执行前

在 `_prepare_inputs()` 相关流程中，如果启用 LoRA：

```python
if self.lora_config:
    self.set_active_loras(
        self.input_batch, num_scheduled_tokens, num_sampled_tokens
    )
```

位置：`gpu_model_runner.py:2193` 到 `gpu_model_runner.py:2201`

`set_active_loras()` 会：

```text
1. InputBatch.make_lora_inputs() 生成 prompt_lora_mapping / token_lora_mapping；
2. 构造 LoRAMapping；
3. WorkerLoRAManager.set_active_adapters()；
4. 必要时加载 adapter；
5. 激活 adapter 到 GPU slot；
6. 更新 punica wrapper metadata。
```

对应源码：

```text
gpu_input_batch.py:976 到 gpu_input_batch.py:999
lora_model_runner_mixin.py:73 到 lora_model_runner_mixin.py:91
worker_manager.py:183 到 worker_manager.py:219
model_manager.py:285 到 model_manager.py:367
```

### 6.4 forward 阶段

```text
model forward
  → LoRA-wrapped Linear / Embedding / LM head
  → base layer output
  → punica wrapper 根据 token_lora_mapping 选择 LoRA slot
  → 加上 LoRA delta
  → 后续 logits / pooling / sampling
```

Linear 的典型源码：

```text
base_linear.py:195 到 base_linear.py:220
```

---

## 7. LoRA 不负责什么

### 7.1 不负责调度 token budget

Scheduler 决定：

```text
哪些请求本轮执行；
每个请求执行多少 token；
KV block 如何分配；
prefill / decode 如何推进。
```

LoRA 只影响这些 token 在模型 forward 时使用哪组增量权重。

`max_loras` 限制的是：

```text
单个 batch 中最多多少个 distinct LoRA adapter。
```

不是 token budget，也不是 request 数量上限。

### 7.2 不直接改变 sampling 规则

LoRA 会改变模型 hidden states / logits 的来源，因为权重 delta 被叠加到了 forward 里。

但它不直接改变：

```text
temperature
top_p
top_k
presence penalty
frequency penalty
stop tokens
structured output grammar
```

这些仍由 `SamplingParams`、sampler、grammar manager 等模块控制。

可以理解为：

```text
LoRA 改变“模型算出来的 logits”；
sampling 改变“如何从 logits 选 token”。
```

### 7.3 不直接改变 decoder KV cache 分配方式

LoRA 不负责 KV block 分配、prefix cache 命中或 KV connector。

但是要注意语义影响：

```text
同一段 prompt，如果使用不同 LoRA，模型计算结果不同；
因此 prefix cache / KV cache 的正确性必须考虑模型权重相关因素。
```

从职责上说：

```text
KV cache manager 管 KV block；
LoRA manager 管 adapter 权重和 active mapping。
```

### 7.4 不是 tokenizer 或 prompt template

LoRARequest 不等于 prompt template，也不替代 tokenizer。

同一个 adapter 可能配套某种 prompt 格式，但在 vLLM 内部：

```text
prompt tokenization / chat template：输入处理层职责；
LoRA adapter：模型权重增量职责。
```

### 7.5 不是完整模型热切换

LoRA 不加载一份新 base model；它复用当前模型结构。

如果 adapter 和 base model 不匹配，应该在加载 / 校验阶段失败，而不是把 base model 换成 adapter 声称的模型。

---

## 8. LoRA 和量化的关系

LoRA 可以和量化 base model 共存，但职责不同。

在 Linear LoRA wrapper 中，base output 仍通过 base layer 的 quant method 执行：

```python
output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
```

位置：`base_linear.py:195` 到 `base_linear.py:199`

LoRA delta 再通过 LoRA 权重和 punica wrapper 叠加：

```python
self.punica_wrapper.add_lora_linear(...)
```

位置：`base_linear.py:218` 到 `base_linear.py:220`

所以边界是：

```text
量化：
  base model 权重如何存储和执行。

LoRA：
  adapter delta 如何加载、激活和叠加。
```

LoRA 的 dtype 由 `LoRAConfig.lora_dtype` 控制：

```text
"auto" 时默认使用 base model dtype。
```

位置：`config/lora.py:46` 到 `config/lora.py:47`，`config/lora.py:127` 到 `config/lora.py:131`

---

## 9. LoRA 和多模态 adapter 的关系

普通语言模型 LoRA 主要作用在 language model 模块。

多模态模型还可能支持 tower / connector LoRA：

```python
enable_tower_connector_lora: bool = False
```

位置：`config/lora.py:62` 到 `config/lora.py:66`

`LoRAModelManager._maybe_init_mm()` 会为不同模型部分准备不同 punica wrapper：

```text
language_model wrapper
tower_model wrapper
connector wrapper
```

位置：`model_manager.py:164` 到 `model_manager.py:269`

执行多模态 encoder 时，如果支持 tower/connector LoRA，会基于 encoder input 构造独立 mapping：

```python
tower_mapping = LoRAMapping(..., type=LoRAMappingType.TOWER)
self.lora_manager.set_active_adapters(lora_requests, tower_mapping)
```

位置：`gpu_model_runner.py:2941` 到 `gpu_model_runner.py:2973`

connector 也类似：

```python
connector_mapping = LoRAMapping(..., type=LoRAMappingType.CONNECTOR)
self.lora_manager.set_active_adapters(lora_requests, connector_mapping)
```

位置：`gpu_model_runner.py:2994` 到 `gpu_model_runner.py:3008`

所以多模态 LoRA 的角色可以理解为：

```text
language LoRA：
  作用于语言模型 token forward。

tower / connector LoRA：
  作用于多模态 encoder / projector 等模块。
```

但这个能力是实验特性，并非所有多模态模型都支持。

---

## 10. LoRA 和并行的关系

LoRA 不替代 TP / PP / DP；它必须适配这些并行方式。

从当前代码能看到几个关键点。

### 10.1 Tensor parallel 下需要切 LoRA A/B

`BaseLinearLayerWithLoRA` 提供：

```text
slice_lora_a()
slice_lora_b()
```

位置：`base.py:17` 到 `base.py:39`

`set_lora()` 时如果 `tp_size > 1` 会切分：

```python
if self.tp_size > 1:
    lora_a = self.slice_lora_a(lora_a)
    lora_b = self.slice_lora_b(lora_b)
```

位置：`base_linear.py:173` 到 `base_linear.py:183`

### 10.2 fully_sharded_loras 控制切分方式

`LoRAConfig.fully_sharded_loras` 的注释说明：

```text
默认只对部分 LoRA 计算做 tensor parallel shard；
开启后使用 fully sharded layers；
在高 sequence length / 高 rank / 高 TP size 时可能更快。
```

位置：`config/lora.py:38` 到 `config/lora.py:42`

### 10.3 PP 下只有存在的 layer 会被包装

`LoRAModelManager._create_lora_modules()` 会跳过 `PPMissingLayer`：

```python
if isinstance(module, PPMissingLayer):
    continue
```

位置：`model_manager.py:385` 到 `model_manager.py:387`

这说明 pipeline parallel 下，每个 stage 只对本 rank 实际持有的模块注入 LoRA。

---

## 11. LoRA 生命周期的三个状态

### 11.1 Registered

adapter 已经作为 `LoRAModel` 注册到 manager：

```text
_registered_adapters[lora_id] = LoRAModel
```

位置：`model_manager.py:94`、`model_manager.py:333` 到 `model_manager.py:336`

它表示：

```text
adapter 已加载到 manager 管理范围内，通常在 CPU/cache 侧有权重对象。
```

### 11.2 Active

adapter 已经被放入 GPU LoRA slot：

```python
self.lora_index_to_id[index] = lora_model.id
module.set_lora(index, module_lora.lora_a, module_lora.lora_b)
```

位置：`model_manager.py:285` 到 `model_manager.py:324`

它表示：

```text
这个 adapter 可以被当前 forward 的 LoRAMapping 引用。
```

### 11.3 Mapped

当前 batch 的 token/prompt mapping 已经写入 punica wrapper：

```python
punica_wrapper.update_metadata(
    mapping,
    self.lora_index_to_id,
    self.lora_slots + 1,
    self.vocab_size,
)
```

位置：`model_manager.py:362` 到 `model_manager.py:367`

它表示：

```text
当前这次 forward 中，每个 token 应该用哪个 active LoRA slot。
```

三个状态的关系：

```text
registered：adapter 已被 manager 认识；
active：adapter 权重已进入 LoRA slot；
mapped：本轮 batch 的 token 已指向具体 LoRA slot。
```

---

## 12. LoRA 的限制和边界

### 12.1 lora_int_id 必须大于 0

`LoRARequest.__post_init__()` 会检查：

```python
if self.lora_int_id < 1:
    raise ValueError(...)
```

位置：`request.py:39` 到 `request.py:45`

因为 runtime 中 `0` 用来表示“无 LoRA”。

### 12.2 单 batch LoRA 数不能超过 max_loras

LRU worker manager 会检查：

```python
if len(loras_map) > self._adapter_manager.lora_slots:
    raise RuntimeError(...)
```

位置：`worker_manager.py:258` 到 `worker_manager.py:269`

普通 worker manager 也有类似检查：

位置：`worker_manager.py:194` 到 `worker_manager.py:206`

### 12.3 CPU cache 容量不能小于 GPU LoRA slots

`LoRAConfig` 校验：

```python
if self.max_cpu_loras is None:
    self.max_cpu_loras = self.max_loras
elif self.max_cpu_loras < self.max_loras:
    raise ValueError(...)
```

位置：`config/lora.py:108` 到 `config/lora.py:116`

含义：

```text
CPU 侧至少要能容纳 GPU 同时活跃的 LoRA 数。
```

### 12.4 只支持被 vLLM 识别的 target modules

`LoRAModelManager._match_target_modules()` 先检查模块是否在 vLLM 支持列表里，再检查 `LoRAConfig.target_modules` 限制：

```python
if not is_supported_lora_module(module_name, self.supported_lora_modules):
    return False
return is_in_target_modules(...)
```

位置：`model_manager.py:672` 到 `model_manager.py:692`

如果用户指定了 `target_modules`，但匹配到的模块无法被 LoRA wrapper 替换，会抛错。

位置：`model_manager.py:481` 到 `model_manager.py:496`

---

## 13. 容易疑惑的点

### 13.1 LoRARequest 是否代表 adapter 已经加载？

不是。

`LoRARequest` 只携带 adapter 的 name / id / path。真正加载发生在：

```text
WorkerLoRAManager._load_adapter()
```

位置：`worker_manager.py:99` 到 `worker_manager.py:162`

### 13.2 add_lora 和请求携带 LoRARequest 是一回事吗？

不是。

```text
请求携带 LoRARequest：
  表示这个请求希望使用某个 adapter。

add_lora：
  控制面显式把 adapter 加载到 worker manager。
```

在 LRU manager 下，如果请求需要的 LoRA 尚未加载，执行前也可能自动加载。

### 13.3 LoRA 是否改变 tokenizer？

不改变。

LoRA 可能带 extra vocab 或 embedding/lm_head 相关权重，但 tokenizer 和 chat template 仍是输入处理层职责。

### 13.4 LoRA 是否改变 KV cache？

不直接管理 KV cache。

LoRA 改变 forward 计算，因此从语义上会影响 KV 的内容；但 KV block 的分配、释放、prefix cache 管理仍在 KV cache manager。

### 13.5 无 LoRA 请求在 mapping 中如何表达？

`InputBatch.request_lora_mapping` 中，`0` 表示无 LoRA。

位置：`gpu_input_batch.py:477` 到 `gpu_input_batch.py:479`

LoRARequest 的 `lora_int_id` 必须大于 0，也正是为了把 0 留给 no-LoRA。

### 13.6 LoRA 是否只用于 Linear？

不只。

当前 LoRA layer 目录包含：

```text
base_linear.py
column_parallel_linear.py
row_parallel_linear.py
replicated_linear.py
vocal_parallel_embedding.py
logits_processor.py
fused_moe.py
```

所以它可以作用于 Linear、Embedding、LM head/logits processor、MoE 等被支持模块。

---

## 14. 总结

LoRA 在 vLLM 中不是一个单独的 layer 小功能，而是一条贯穿配置、请求、worker 管理、batch 状态、layer wrapper 和 kernel mapping 的执行链路。

最完整的压缩版是：

```text
LoRAConfig 决定服务是否启用 LoRA 以及资源上限；
LoRARequest 决定某个请求想用哪个 adapter；
WorkerLoRAManager 负责加载和生命周期；
LoRAModelManager 负责把 adapter 放入模型 LoRA slot；
InputBatch 负责 batch 内 request/token 到 lora_id 的映射；
LoRA-wrapped layer 负责执行 base output + LoRA delta；
sampler 仍然只消费最终 logits。
```

如果只记住一句话：

```text
LoRA 是 vLLM 中请求级可切换的增量权重机制；它复用 base model，通过 manager 管理 adapter，通过 token 级 mapping 支持 batch 混合执行，并在 LoRA layer 中把低秩 delta 叠加到模型 forward 上。
```
