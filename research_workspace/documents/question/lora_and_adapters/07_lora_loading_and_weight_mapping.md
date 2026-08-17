# 07. LoRA 权重如何加载和映射？

源码位置：

- `code/vllm/vllm/lora/request.py`
- `code/vllm/vllm/lora/worker_manager.py`
- `code/vllm/vllm/lora/peft_helper.py`
- `code/vllm/vllm/lora/lora_model.py`
- `code/vllm/vllm/lora/lora_weights.py`
- `code/vllm/vllm/lora/model_manager.py`
- `code/vllm/vllm/lora/utils.py`
- `code/vllm/vllm/lora/layers/`
- `code/vllm/vllm/model_executor/models/utils.py`
- `code/vllm/vllm/model_executor/utils.py`

本问题关注：外部 LoRA checkpoint 中的权重，如何经过 PEFT config 校验、权重文件读取、名字解析、target module 校验、packed/fused module 合并、TP/EP 切分，最终放入 vLLM 内部 LoRA layer 的 slot。

---

## 1. 一句话回答

LoRA 权重加载的核心，是把 PEFT checkpoint 中的 `lora_A` / `lora_B` 权重名，映射到 vLLM 已经 LoRA-wrapped 的内部 module 名，并在进入 GPU slot 前处理 rank、alpha/scaling、fused layer、MoE expert、TP/EP 切分和模型自定义命名映射。

主链路是：

```text
LoRARequest(lora_name, lora_int_id, lora_path)
  → WorkerLoRAManager.add_adapter()
  → _load_adapter()
  → get_adapter_absolute_path()
  → PEFTHelper.from_local_dir(adapter_config.json)
  → peft_helper.validate_legal(LoRAConfig)
  → LoRAModel.from_local_checkpoint()
      → 读取 adapter_model.safetensors / bin / pt / tensorizer
      → check_unexpected_modules()
      → parse_fine_tuned_lora_name()
      → LoRALayerWeights.from_config()
      → 填充 lora_a / lora_b tensor
  → LoRAModelManager.add_adapter()
      → _create_merged_loras_inplace()
      → pack fused / MoE LoRA 权重
      → optimize() 合并 scaling
  → activate_adapter()
      → 找 GPU slot
      → module.set_lora(index, lora_a, lora_b)
      → TP 下 slice_lora_a / slice_lora_b
```

最小心智模型：

```text
checkpoint key 是外部训练框架命名；
vLLM module name 是内部执行层命名；
LoRA loader 负责把二者对齐；
LoRAModelManager 负责把对齐后的权重放进 LoRA layer slot。
```

---

## 2. 输入对象：LoRARequest 只提供 adapter 身份和路径

`LoRARequest` 定义在：`request.py:8`

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

它在加载链路里的作用是：

```text
lora_name：
  adapter 名字，用于请求级识别和 LoRARequest eq/hash。

lora_int_id：
  runtime 整数 ID，也就是后续 LoRAModel.id 和 LoRA slot mapping 的 adapter_id。

lora_path：
  adapter checkpoint 的路径或 HF / ModelScope repo id。

tensorizer_config_dict：
  如果 adapter 用 tensorizer 格式存储，从这里拿 tensorizer 读取配置。

load_inplace：
  强制重新加载同 id adapter，用新权重替换旧权重。

is_3d_lora_weight：
  MoE LoRA checkpoint 是否是 3D fused 格式，供 mixed MoE LoRA 路径判断。
```

`LoRARequest.__post_init__()` 要求：

```text
lora_int_id > 0；
lora_path 不能为空。
```

位置：`request.py:39` 到 `request.py:45`

这里要注意：

```text
0 在 runtime mapping 中通常表示 no-LoRA；
所以真实 adapter 的 lora_int_id 必须从 1 开始。
```

---

## 3. WorkerLoRAManager 负责把请求变成 LoRAModel

入口：`worker_manager.py:273`

LRU worker manager 的 `add_adapter()` 流程是：

```python
if lora_request.lora_int_id not in self.list_adapters() or lora_request.load_inplace:
    lora = self._load_adapter(lora_request)
    self._adapter_manager.remove_adapter(lora.id)
    if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
        self._adapter_manager.remove_oldest_adapter()
    loaded = self._adapter_manager.add_adapter(lora)
else:
    loaded = self._adapter_manager.get_adapter(lora_request.lora_int_id) is not None
self._adapter_manager.activate_adapter(lora_request.lora_int_id)
```

位置：`worker_manager.py:273` 到 `worker_manager.py:307`

这说明：

```text
1. 如果 adapter 未加载，先从 checkpoint 读取并构造 LoRAModel；
2. 如果 load_inplace=True，即使同 id 已存在也重新加载；
3. 先验证新 adapter 可加载，再淘汰旧 adapter；
4. CPU cache 超容量时淘汰 oldest；
5. 最后激活 adapter 到 GPU slot。
```

普通 `WorkerLoRAManager` 路径也类似，只是没有 LRU 自动淘汰：

位置：`worker_manager.py:194` 到 `worker_manager.py:219`

---

## 4. 路径解析：本地路径、HF、ModelScope

`_load_adapter()` 先解析路径：

```python
lora_path = get_adapter_absolute_path(lora_request.lora_path)
```

位置：`worker_manager.py:112`

`get_adapter_absolute_path()` 逻辑：

```text
如果 lora_path 是绝对路径：
  原样返回。

如果以 ~ 开头：
  expanduser。

如果相对路径在本地存在：
  转成 abspath。

否则：
  按配置尝试从 ModelScope 或 Hugging Face Hub snapshot_download。

下载失败：
  记录异常，返回原始路径；后续读取 checkpoint 时再失败。
```

位置：`utils.py:303` 到 `utils.py:357`

所以 `lora_path` 支持：

```text
- 本地绝对路径；
- 本地相对路径；
- ~ 用户目录路径；
- Hugging Face repo id；
- ModelScope repo id。
```

---

## 5. adapter_config.json 如何读取和校验

### 5.1 PEFTHelper 读取配置

`WorkerLoRAManager._load_adapter()` 中：

```python
peft_helper = PEFTHelper.from_local_dir(
    lora_path,
    self.max_position_embeddings,
    lora_request.tensorizer_config_dict,
)
```

位置：`worker_manager.py:114` 到 `worker_manager.py:118`

`PEFTHelper.from_local_dir()` 会读取：

```text
adapter_config.json
```

位置：`peft_helper.py:81` 到 `peft_helper.py:112`

如果是 tensorizer，则从 tensorizer_dir 读取对应 config。

### 5.2 PEFTHelper 关心哪些字段

`PEFTHelper` 必需字段：

```python
r: int
lora_alpha: int
target_modules: list[str] | str
```

位置：`peft_helper.py:27` 到 `peft_helper.py:30`

其他字段：

```python
bias: Literal["none"] = "none"
modules_to_save: list[str] | None = None
use_rslora: bool = False
use_dora: bool = False
vllm_lora_scaling_factor: float = 1.0
vllm_max_position_embeddings: int | None = False
```

位置：`peft_helper.py:32` 到 `peft_helper.py:40`

### 5.3 scaling 如何计算

`PEFTHelper.__post_init__()` 中：

```python
if self.use_rslora:
    self.vllm_lora_scaling_factor = self.lora_alpha / math.sqrt(self.r)
else:
    self.vllm_lora_scaling_factor = self.lora_alpha / self.r
```

位置：`peft_helper.py:53` 到 `peft_helper.py:59`

后面 `LoRALayerWeights.from_config()` 会把这个 scaling 写入权重对象：

```python
return cls(
    module_name,
    peft_helper.r,
    peft_helper.lora_alpha,
    None,
    None,
    peft_helper.vllm_lora_scaling_factor,
)
```

位置：`lora_weights.py:57` 到 `lora_weights.py:70`

### 5.4 合法性校验

`_load_adapter()` 会调用：

```python
peft_helper.validate_legal(self.lora_config)
```

位置：`worker_manager.py:120` 到 `worker_manager.py:122`

校验内容：

```text
modules_to_save 必须为 None；
DoRA 不支持；
r 不能大于 LoRAConfig.max_lora_rank；
bias 必须为 none。
```

位置：`peft_helper.py:42` 到 `peft_helper.py:51`，`peft_helper.py:114` 到 `peft_helper.py:128`

这说明当前 vLLM LoRA loader 对 PEFT 特性是有明确边界的。

---

## 6. expected_lora_modules 如何生成

`WorkerLoRAManager._load_adapter()` 会从 `LoRAModelManager` 拿到：

```python
supported_lora_modules = self._adapter_manager.supported_lora_modules
packed_modules_mapping = self._adapter_manager.packed_modules_mapping
```

位置：`worker_manager.py:100` 到 `worker_manager.py:102`

然后构造 expected module 列表：

```python
expected_lora_lst = []
for module in supported_lora_modules:
    if module in packed_modules_mapping:
        expected_lora_lst.extend(packed_modules_mapping[module])
    else:
        expected_lora_lst.append(module)
    if module == "experts":
        expected_lora_lst.append(module)
expected_lora_modules = set(expected_lora_lst)
```

位置：`worker_manager.py:103` 到 `worker_manager.py:111`

含义：

```text
vLLM 内部可能有 fused runtime module，比如 qkv_proj / gate_up_proj / experts；
checkpoint 里可能按 unfused 子模块保存，比如 q_proj / k_proj / v_proj / gate_proj / up_proj；
expected_lora_modules 要接受这些 adapter-visible 子模块名。
```

`packed_modules_mapping` 来自：

```python
process_packed_modules_mapping(self.model, force_2d_moe=...)
```

位置：`model_manager.py:125` 到 `model_manager.py:127`

`process_packed_modules_mapping()` 会处理普通 packed modules 和 MoE experts：

位置：`utils.py:360` 到 `utils.py:392`

---

## 7. checkpoint 文件如何读取

`LoRAModel.from_local_checkpoint()` 支持几种文件：

```python
adapter_model.safetensors
adapter_model.bin
adapter_model.pt
adapter_model.tensors  # tensorizer
```

位置：`lora_model.py:205` 到 `lora_model.py:207`，`lora_model.py:244` 到 `lora_model.py:295`

### 7.1 safetensors

如果存在 `adapter_model.safetensors`：

```text
1. safe_open；
2. 用 key 做 unexpected module 校验；
3. 遍历 f.keys()；
4. 如开启 EP MoE slicing，跳过 remote expert key；
5. f.get_tensor(module) 读取 tensor。
```

位置：`lora_model.py:260` 到 `lora_model.py:277`

### 7.2 bin / pt

如果是 `adapter_model.bin` 或 `adapter_model.pt`：

```python
tensors = torch.load(lora_file_path, map_location=device, weights_only=True)
check_unexpected_modules(tensors)
```

位置：`lora_model.py:277` 到 `lora_model.py:284`

如果有 `moe_ep_spec`，会在 dict 里过滤 remote expert key：

位置：`lora_model.py:285` 到 `lora_model.py:293`

### 7.3 tensorizer

如果有 `tensorizer_config_dict`：

```text
1. 构造 TensorizerConfig；
2. 路径指向 adapter_model.tensors；
3. TensorDeserializer 读取；
4. check_unexpected_modules。
```

位置：`lora_model.py:244` 到 `lora_model.py:258`

---

## 8. checkpoint key 如何解析成 vLLM module name

核心函数：`parse_fine_tuned_lora_name()`。

位置：`utils.py:155`

它处理 PEFT 常见命名：

```text
base_model.model.xxx.lora_A.weight
base_model.model.xxx.lora_B.weight
xxx.lora_A.weight
xxx.lora_B.weight
xxx.lora_embedding_A
xxx.lora_embedding_B
```

核心逻辑：

```python
if name.startswith("base_model.model."):
    name = name.replace("base_model.model.", "")
    name = weights_mapper._map_name(name) if weights_mapper else name
    name = "base_model.model." + name
else:
    name = weights_mapper._map_name(name) if weights_mapper else name

start_index = 2 if name.startswith("base_model.model.") else 0
parts = name.split(".")
if parts[-1] == "weight" and (parts[-2] == "lora_A" or parts[-2] == "lora_B"):
    new_name = ".".join(parts[start_index:-2])
    return new_name, parts[-2] == "lora_A"

if parts[-1] == "lora_embedding_A" or parts[-1] == "lora_embedding_B":
    new_name = ".".join(parts[start_index:-1])
    return new_name, parts[-1] == "lora_embedding_A"
```

位置：`utils.py:171` 到 `utils.py:196`

返回值是：

```text
(module_name, is_lora_a)
```

例如：

```text
base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
  → module_name = model.layers.0.self_attn.q_proj
  → is_lora_a = True

base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
  → module_name = model.layers.0.self_attn.q_proj
  → is_lora_a = False
```

如果模型提供 `hf_to_vllm_mapper`，会先映射名字：

```python
hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
```

位置：`worker_manager.py:126` 到 `worker_manager.py:127`

这对 Qwen-VL 等 HF checkpoint 命名和 vLLM 内部命名不一致的模型很重要。

---

## 9. unexpected module 如何报错

`from_local_checkpoint()` 内部定义 `check_unexpected_modules()`。

核心规则：

```text
跳过 base embedding weights；
跳过包含 base_layer 的 PEFT 特殊 key；
跳过模型指定 skip_prefixes；
解析 module_name；
如果是 experts，按 expert suffix 检查 expected_lora_modules；
否则检查 module_name 最后一段是否在 expected_lora_modules；
不符合就加入 unexpected_modules；
最后抛 ValueError。
```

位置：`lora_model.py:212` 到 `lora_model.py:242`

错误大意是：

```text
expected target modules in {...} but received [...]
Please verify that the loaded LoRA module is correct
```

这说明 vLLM 不会默默加载完全不匹配的 adapter；checkpoint 的 target module 必须能对齐当前 base model 支持的 LoRA module。

---

## 10. LoRAModel.from_lora_tensors 如何组装权重

读取 tensors 后：

```python
return cls.from_lora_tensors(...)
```

位置：`lora_model.py:297` 到 `lora_model.py:306`

`from_lora_tensors()` 的核心循环：

```python
for tensor_name, tensor in tensors.items():
    if is_base_embedding_weights(tensor_name):
        continue
    if skip_prefixes and cls._should_skip_module(tensor_name, skip_prefixes):
        continue
    module_name, is_lora_a = parse_fine_tuned_lora_name(tensor_name, weights_mapper)
    if module_name not in loras:
        loras[module_name] = LoRALayerWeights.from_config(module_name, peft_helper)
    if is_lora_a:
        loras[module_name].lora_a = tensor.to(device=device, dtype=dtype)
    else:
        loras[module_name].lora_b = tensor.to(device=device, dtype=dtype)
```

位置：`lora_model.py:117` 到 `lora_model.py:164`

这里有几个关键点。

### 10.1 先按 config 建 LoRALayerWeights

```python
LoRALayerWeights.from_config(module_name, peft_helper)
```

位置：`lora_model.py:140` 到 `lora_model.py:143`

`from_config()` 先填：

```text
rank = peft_helper.r
lora_alpha = peft_helper.lora_alpha
scaling = peft_helper.vllm_lora_scaling_factor
lora_a = None
lora_b = None
```

位置：`lora_weights.py:57` 到 `lora_weights.py:70`

然后再按 tensor name 填 A/B。

### 10.2 dtype 在加载时转换

```python
tensor.to(device=device, dtype=dtype)
```

位置：`lora_model.py:155`、`lora_model.py:159`

`dtype` 来自 `LoRAConfig.lora_dtype`。

`LoRAConfig.verify_with_model_config()` 中：

```python
if self.lora_dtype in (None, "auto"):
    self.lora_dtype = model_config.dtype
elif isinstance(self.lora_dtype, str):
    self.lora_dtype = getattr(torch, self.lora_dtype)
```

位置：`config/lora.py:127` 到 `config/lora.py:131`

### 10.3 embedding LoRA 要校验 vocab size

如果 key 是 `lora_embedding_A`，并且提供了 `model_vocab_size`：

```python
if model_vocab_size is not None and model_vocab_size != tensor.shape[1]:
    raise RuntimeError(...)
```

位置：`lora_model.py:146` 到 `lora_model.py:154`

说明 embedding LoRA 不能和 base model vocab size 不一致。

### 10.4 base embedding weights 被跳过

```python
if is_base_embedding_weights(tensor_name):
    continue
```

位置：`lora_model.py:131` 到 `lora_model.py:133`

`is_base_embedding_weights()` 硬编码跳过：

```text
.embed_tokens.base_layer.weight
.lm_head.base_layer.weight
```

位置：`utils.py:199` 到 `utils.py:205`

---

## 11. LoRALayerWeights 如何保存 rank / alpha / scaling

`LoRALayerWeights` 表示单个 module 的 LoRA A/B：

```python
module_name: str
rank: int
lora_alpha: int
lora_a: torch.Tensor
lora_b: torch.Tensor
scaling: float
```

位置：`lora_weights.py:13` 到 `lora_weights.py:35`

如果 scaling 没传，则默认：

```text
scaling = lora_alpha / rank
```

位置：`lora_weights.py:31` 到 `lora_weights.py:35`

`optimize()` 会把 scaling 合进 `lora_b`：

```python
if self.scaling == 1:
    return self
self.lora_b *= self.scaling
self.scaling = 1
```

位置：`lora_weights.py:36` 到 `lora_weights.py:42`

这样后续执行时可以少处理一个 scaling 乘法。

---

## 12. LoRAModelManager 如何包装模型层

LoRA 权重能加载成功，还要模型层已经被替换成 LoRA wrapper。

`LoRAModelManager.__init__()` 会：

```text
1. get_supported_lora_modules(model)；
2. process_packed_modules_mapping(model)；
3. 初始化 punica wrapper；
4. _create_lora_modules() 替换模型 module。
```

位置：`model_manager.py:88` 到 `model_manager.py:137`

### 12.1 supported_lora_modules

`get_supported_lora_modules()` 会收集：

```text
- 模型声明的 embedding_modules；
- 所有 LinearBase 的最后一段名字；
- 所有 MoERunner 的最后一段名字。
```

位置：`utils.py:208` 到 `utils.py:229`

### 12.2 target_modules 过滤

`_match_target_modules()` 先检查 module 是否支持 LoRA，再检查部署时 `LoRAConfig.target_modules`：

```python
if not is_supported_lora_module(module_name, self.supported_lora_modules):
    return False
return is_in_target_modules(...)
```

位置：`model_manager.py:672` 到 `model_manager.py:692`

`is_in_target_modules()` 支持：

```text
target_modules=None：全部支持模块通过；
模块 suffix 命中 target_modules；
完整 module_name 命中 target_modules；
packed parent/child 互相匹配。
```

位置：`utils.py:260` 到 `utils.py:300`

### 12.3 替换 module

`_create_lora_modules()` 遍历模型：

```python
for module_name, module in self.model.named_modules(remove_duplicate=False):
    if isinstance(module, PPMissingLayer):
        continue
    if not self._match_target_modules(module_name):
        continue
    new_module = replace_submodule(
        self.model,
        module_name,
        from_layer(...),
    )
```

位置：`model_manager.py:375` 到 `model_manager.py:502`

`from_layer()` 会按 `_all_lora_classes` 顺序选择第一个能替换的 wrapper：

位置：`utils.py:76` 到 `utils.py:124`

可替换类型包括：

```text
VocabParallelEmbeddingWithLoRA
ColumnParallelLinearWithLoRA
MergedColumnParallelLinearWithLoRA
QKVParallelLinearWithLoRA
MergedQKVParallelLinearWithLoRA
RowParallelLinearWithLoRA
ReplicatedLinearWithLoRA
LogitsProcessorWithLoRA
各种 fully sharded LoRA wrapper
FusedMoEWithLoRA / FusedMoE3DWithLoRA
```

位置：`utils.py:76` 到 `utils.py:95`

---

## 13. fused / packed layer 如何映射

很多 vLLM runtime layer 是 fused 的，例如：

```text
q_proj + k_proj + v_proj → qkv_proj
gate_proj + up_proj → gate_up_proj
MoE experts.N.w1/w2/w3 → experts
```

checkpoint 可能保存 unfused 子模块。vLLM 通过 `packed_modules_mapping` 做对齐。

### 13.1 packed_modules_mapping 的来源

```python
self.packed_modules_mapping = process_packed_modules_mapping(
    self.model, force_2d_moe=self._enable_mixed_moe_lora_format
)
```

位置：`model_manager.py:125` 到 `model_manager.py:127`

普通模型走：

```python
return get_packed_modules_mapping(model)
```

位置：`utils.py:391` 到 `utils.py:392`

MoE 模型会额外展开 expert 映射：

位置：`utils.py:360` 到 `utils.py:390`

### 13.2 添加 adapter 时先合并 packed LoRA

`LoRAModelManager._add_adapter()` 会先调用：

```python
self._create_merged_loras_inplace(lora)
```

位置：`model_manager.py:333` 到 `model_manager.py:336`

`_create_merged_loras_inplace()` 遍历 packed modules：

```text
1. 对每个 fused runtime module，找到对应的 checkpoint 子模块 LoRA；
2. 缺失的子模块用 None 占位；
3. 普通 packed layer 用 PackedLoRALayerWeights.pack()；
4. MoE experts 用 PackedLoRALayerWeights.pack_moe()；
5. 删除原始子模块 LoRA，保留 packed 后的 LoRA。
```

位置：`model_manager.py:724` 到 `model_manager.py:816`

### 13.3 PackedLoRALayerWeights

`PackedLoRALayerWeights.pack()` 会把多个子 LoRA 合成一个 packed LoRA：

```python
obj = cls(
    module_name,
    rank,
    [lora.lora_alpha if lora is not None else None for lora in loras],
    [lora.lora_a if lora is not None else None for lora in loras],
    [lora.lora_b if lora is not None else None for lora in loras],
    scaling=[1 if lora is not None else None for lora in loras],
)
```

位置：`lora_weights.py:126` 到 `lora_weights.py:152`

`pack_moe()` 会把 expert 的 w1/w2/w3 LoRA stack 起来：

位置：`lora_weights.py:154` 到 `lora_weights.py:228`

---

## 14. MoE / Expert Parallel 特殊处理

### 14.1 读取阶段跳过非本地 expert

`MoEEPLoadSpec` 保存：

```python
ep_rank: int
local_num_experts: int
global_num_experts: int
```

位置：`lora_model.py:25` 到 `lora_model.py:35`

如果启用 expert parallel，`from_local_checkpoint()` 会在读取 safetensors key 时跳过远端 expert：

```python
if moe_ep_spec is not None and _is_remote_expert_key(module, moe_ep_spec):
    continue
```

位置：`lora_model.py:271` 到 `lora_model.py:276`

对于 `.bin/.pt`，先读入 dict，再过滤：

位置：`lora_model.py:285` 到 `lora_model.py:293`

### 14.2 manager 阶段限制本 rank experts

`_create_merged_loras_inplace()` 中，如果 module name 是 `.experts`，会限制到本 rank local experts：

位置：`model_manager.py:724` 到 `model_manager.py:733`

后面 `_stack_moe_lora_weights()` 也会按 `ep_rank/local_num_experts/global_num_experts` reshape 和 slice：

位置：`model_manager.py:817` 到 `model_manager.py:880`

### 14.3 mixed MoE LoRA format

`LoRAConfig.enable_mixed_moe_lora_format`：

```text
允许 2D-format 和 3D-format MoE LoRA adapters 在同一部署中服务。
```

位置：`config/lora.py:74` 到 `config/lora.py:79`

如果启用，`LoRAModelManager` 会强制使用 universal 2D wrapper，并在需要时做 3D→2D 转换：

位置：`model_manager.py:114` 到 `model_manager.py:127`，`model_manager.py:779` 到 `model_manager.py:789`

`LoRARequest.is_3d_lora_weight` 会被写到 LoRAModel：

```python
lora.is_3d_lora_weight = lora_request.is_3d_lora_weight
```

位置：`worker_manager.py:145` 到 `worker_manager.py:149`

---

## 15. activate_adapter 如何把 LoRA 权重放入 GPU slot

加载好的 `LoRAModel` 还只是 adapter 权重集合；执行前要激活到 slot。

`LoRAModelManager.activate_adapter()`：

```text
1. 如果已经 active，返回 False；
2. 在 lora_index_to_id 找第一个空 slot；
3. active_adapters[lora_id] = None；
4. lora_index_to_id[index] = lora_model.id；
5. 遍历每个 LoRA-wrapped module；
6. 找到这个 module 对应的 LoRALayerWeights；
7. module.set_lora(index, lora_a, lora_b)；
8. 如果没有对应权重，module.reset_lora(index)。
```

位置：`model_manager.py:285` 到 `model_manager.py:324`

这一步才真正把 adapter 权重复制到 LoRA layer 预分配的 stacked tensor：

```text
lora_a_stacked[index]
lora_b_stacked[index]
```

---

## 16. TP 下权重如何切分

LoRA layer 在 `set_lora()` 中根据 TP 做切分。

### 16.1 ColumnParallelLinear

Column parallel 下，通常切 `lora_b` 的 output dim：

```python
shard_size = self.output_size
start_idx = self.tp_rank * shard_size
end_idx = (self.tp_rank + 1) * shard_size
lora_b = lora_b[start_idx:end_idx, :]
```

位置：`column_parallel_linear.py:105` 到 `column_parallel_linear.py:128`

Merged column parallel 会按左右半区或 output slices 切。

位置：`column_parallel_linear.py:246` 到 `column_parallel_linear.py:326`

### 16.2 QKVParallelLinear

QKV 切分需要分别处理 Q/K/V：

```text
Q 按 q heads 切；
K/V 按 kv heads 和 num_kv_head_replicas 切；
最后 cat 成当前 rank 的 lora_b。
```

位置：`column_parallel_linear.py:365` 到 `column_parallel_linear.py:428`

### 16.3 RowParallelLinear

Row parallel 下，通常切 `lora_a` 的 input dim：

```python
shard_size = self.input_size
start_idx = self.tp_rank * shard_size
end_idx = (self.tp_rank + 1) * shard_size
lora_a = lora_a[:, start_idx:end_idx]
```

位置：`row_parallel_linear.py:22` 到 `row_parallel_linear.py:40`

Fully sharded RowParallel 还会切 `lora_b`：

位置：`row_parallel_linear.py:101` 到 `row_parallel_linear.py:177`

---

## 17. embedding / lm_head 如何处理

LoRA 支持 embedding 和 logits processor / lm_head 路径。

`get_supported_lora_modules()` 会读取模型的 `embedding_modules`：

位置：`utils.py:213` 到 `utils.py:220`

`from_layer()` 支持：

```text
VocabParallelEmbeddingWithLoRA
LogitsProcessorWithLoRA
```

位置：`utils.py:76` 到 `utils.py:95`

`_create_lora_modules()` 遇到 `lm_head` 时，还会包装 logits processor：

```python
new_module = replace_submodule(
    self.model,
    logits_processor_module_name,
    from_layer_logits_processor(...),
)
```

位置：`model_manager.py:456` 到 `model_manager.py:480`

加载 embedding LoRA 时会校验 vocab size：

位置：`lora_model.py:146` 到 `lora_model.py:154`

这说明 embedding / lm_head LoRA 不是普通 Linear 的简单拷贝，需要处理 vocab parallel 和 logits processor 的输出路径。

---

## 18. skip_prefixes 和模型自定义命名

某些模型会定义不应加载 LoRA 的前缀，例如 MTP 层。

`WorkerLoRAManager._load_adapter()`：

```python
lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None)
```

位置：`worker_manager.py:129` 到 `worker_manager.py:130`

`LoRAModel.from_lora_tensors()` 和 `check_unexpected_modules()` 都会跳过这些 prefix：

位置：`lora_model.py:134` 到 `lora_model.py:136`，`lora_model.py:220` 到 `lora_model.py:224`

命名映射由模型的 `hf_to_vllm_mapper` 处理：

位置：`worker_manager.py:126` 到 `worker_manager.py:127`，`utils.py:171` 到 `utils.py:180`

这解决的问题是：

```text
HF checkpoint 名字和 vLLM 内部 module 名字不完全一致。
```

---

## 19. 常见失败点

### 19.1 adapter_config.json 缺字段

`PEFTHelper.from_dict()` 会检查 `r / lora_alpha / target_modules` 必需字段。

位置：`peft_helper.py:60` 到 `peft_helper.py:78`

### 19.2 rank 超过 max_lora_rank

```text
LoRA rank r > LoRAConfig.max_lora_rank
```

会在 `validate_legal()` 报错。

位置：`peft_helper.py:120` 到 `peft_helper.py:124`

### 19.3 bias / DoRA / modules_to_save 不支持

同样在 `validate_legal()` 报错。

位置：`peft_helper.py:42` 到 `peft_helper.py:51`，`peft_helper.py:125` 到 `peft_helper.py:128`

### 19.4 checkpoint 没有权重文件

如果没有：

```text
adapter_model.safetensors
adapter_model.bin
adapter_model.pt
adapter_model.tensors
```

会抛：

```text
ValueError: <lora_dir> doesn't contain tensors
```

位置：`lora_model.py:294` 到 `lora_model.py:295`

### 19.5 target modules 不匹配

`check_unexpected_modules()` 会抛：

```text
expected target modules in ... but received ...
```

位置：`lora_model.py:236` 到 `lora_model.py:242`

### 19.6 module 匹配到了但不能被 wrapper 替换

如果用户显式指定 `target_modules`，但 vLLM 找不到可用 wrapper，会抛错：

位置：`model_manager.py:481` 到 `model_manager.py:496`

---

## 20. 最小心智模型

LoRA 权重加载可以拆成四个阶段：

```text
阶段 1：读配置
  adapter_config.json → PEFTHelper(r, alpha, target_modules, scaling)

阶段 2：读权重
  adapter_model.safetensors/bin/pt/tensors → tensor dict

阶段 3：名字映射
  PEFT key → parse_fine_tuned_lora_name() → vLLM module name
  hf_to_vllm_mapper / packed_modules_mapping / skip_prefixes 参与修正

阶段 4：进入执行层
  LoRALayerWeights / PackedLoRALayerWeights
    → LoRAModel
    → LoRAModelManager.add_adapter()
    → activate_adapter()
    → module.set_lora(slot, A, B)
```

如果只记住一句话：

```text
vLLM 加载 LoRA 的关键不是简单读 tensor，而是把 PEFT checkpoint 的外部命名和 vLLM 内部 fused、parallel、MoE、embedding/lm_head module 对齐，然后把处理后的 A/B 权重复制进 LoRA layer 的运行时 slot。
```
