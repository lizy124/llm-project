# 11. LoRA 有哪些限制和调试入口？

源码位置：

- `code/vllm/vllm/config/lora.py`
- `code/vllm/vllm/config/compilation.py`
- `code/vllm/vllm/lora/request.py`
- `code/vllm/vllm/lora/peft_helper.py`
- `code/vllm/vllm/lora/lora_model.py`
- `code/vllm/vllm/lora/model_manager.py`
- `code/vllm/vllm/lora/worker_manager.py`
- `code/vllm/vllm/lora/utils.py`
- `code/vllm/vllm/lora/layers/`
- `code/vllm/vllm/v1/engine/input_processor.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/executor/abstract.py`
- `code/vllm/vllm/model_executor/layers/quantization/`

本问题关注：vLLM V1 中 LoRA 常见限制、典型报错位置、fallback / skip 行为，以及如何按“请求 → 调度 → manager → layer → mapping → forward”的链路定位问题。

---

## 1. 一句话回答

LoRA 问题通常不是单点问题，而是链路上某一层契约不满足：

```text
请求层：LoRARequest 是否合法，LoRA 是否启用；
配置层：rank / max_loras / max_cpu_loras / target_modules 是否匹配；
加载层：adapter_config.json 和权重文件是否存在、格式是否支持；
模型层：base model 是否 supports_lora，目标模块是否能被 LoRA wrapper 替换；
调度层：本轮不同 LoRA 数是否超过 max_loras；
执行层：adapter 是否 registered / active，LoRAMapping 是否正确；
后端层：MoE、量化、CUDA graph、multimodal tower/connector 是否走到支持路径。
```

调试时最有效的顺序是：

```text
LoRARequest
  → InputProcessor._validate_lora()
  → Scheduler max_loras 约束
  → Worker / ModelRunner add_lora()
  → WorkerLoRAManager._load_adapter()
  → LoRAModel.from_local_checkpoint()
  → LoRAModelManager._create_lora_modules()
  → activate_adapter()
  → InputBatch.make_lora_inputs()
  → LoRAMapping / Punica wrapper metadata
  → forward / CUDA graph / quant backend
```

---

## 2. 先按阶段划分 LoRA 限制

### 2.1 请求和启用阶段

常见问题：

```text
- 请求传了 LoRARequest，但服务没有启用 LoRA；
- LoRARequest.lora_int_id <= 0；
- LoRARequest.lora_path 为空；
- 多 worker 之间 LoRA 集合不一致；
- 使用 LoRA 自带 tokenizer 的旧用法已经 deprecated。
```

对应入口：

- `code/vllm/vllm/lora/request.py:39` 到 `request.py:45`
- `code/vllm/vllm/v1/engine/input_processor.py:146` 到 `input_processor.py:163`
- `code/vllm/vllm/v1/executor/abstract.py:292` 到 `abstract.py:308`

### 2.2 配置阶段

常见问题：

```text
- adapter rank 超过 max_lora_rank；
- max_cpu_loras < max_loras；
- target_modules 限制过窄，导致期望模块没有被注入；
- enable_tower_connector_lora 开启但模型不支持 tower / connector LoRA；
- fully_sharded_loras 和部分运行选项不兼容；
- LoRA dtype 和 base model dtype / 权重 dtype 预期不一致。
```

对应入口：

- `code/vllm/vllm/config/lora.py:30` 到 `lora.py:80`
- `code/vllm/vllm/config/lora.py:108` 到 `lora.py:132`
- `code/vllm/vllm/lora/peft_helper.py:114` 到 `peft_helper.py:128`

### 2.3 adapter 加载阶段

常见问题：

```text
- adapter_config.json 缺失或字段不完整；
- adapter_model.safetensors / bin / pt / tensorizer 文件缺失；
- checkpoint 中的 module 名和当前 base model 不匹配；
- checkpoint 中出现 vLLM 不支持的 LoRA weight name；
- embedding LoRA vocab size 和 base model vocab size 不一致；
- HuggingFace / ModelScope 下载失败后路径仍不可用。
```

对应入口：

- `code/vllm/vllm/lora/peft_helper.py:80` 到 `peft_helper.py:112`
- `code/vllm/vllm/lora/lora_model.py:167` 到 `lora_model.py:306`
- `code/vllm/vllm/lora/utils.py:303` 到 `utils.py:357`
- `code/vllm/vllm/lora/worker_manager.py:99` 到 `worker_manager.py:162`

### 2.4 模型注入阶段

常见问题：

```text
- base model 不支持 LoRA；
- 模型中没有 supported LoRA modules；
- target module 不是 vLLM 支持的 Linear / Embedding / MoE wrapper；
- target_modules 和 packed module 映射不匹配；
- MoE 模型没有实现 get_expert_mapping；
- 某些 matched module 无法被任何 BaseLayerWithLoRA 子类替换。
```

对应入口：

- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:31` 到 `lora_model_runner_mixin.py:46`
- `code/vllm/vllm/lora/model_manager.py:88` 到 `model_manager.py:137`
- `code/vllm/vllm/lora/model_manager.py:375` 到 `model_manager.py:503`
- `code/vllm/vllm/lora/utils.py:106` 到 `utils.py:124`
- `code/vllm/vllm/lora/utils.py:208` 到 `utils.py:300`
- `code/vllm/vllm/lora/utils.py:360` 到 `utils.py:392`

### 2.5 调度和 batch 阶段

常见问题：

```text
- 单轮 batch 中不同 LoRA 数超过 max_loras，请求被跳过等待；
- InputBatch 中 request_lora_mapping 和 request 生命周期不同步；
- lora_id=0 被误解为有效 adapter；
- LoRARequest id 重复但 path/name 指向不同 adapter，导致复用错误；
- load_inplace 没有使用，导致同 id adapter 没有重新加载。
```

对应入口：

- `code/vllm/vllm/v1/core/sched/scheduler.py:614` 到 `scheduler.py:664`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:244` 到 `gpu_input_batch.py:247`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:468` 到 `gpu_input_batch.py:479`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:530` 到 `gpu_input_batch.py:538`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:976` 到 `gpu_input_batch.py:999`

### 2.6 执行和后端阶段

常见问题：

```text
- adapter registered 但没有 active 到 GPU slot；
- GPU active slots 已满；
- LoRAMapping 没更新或映射到错误 slot；
- pin 太多 adapter，LRU 无法淘汰；
- CUDA graph 没覆盖当前 LoRA active 状态；
- MoE + 量化 + LoRA 组合缺少 backend；
- multimodal tower / connector LoRA 的 mm hash / mapping 不正确。
```

对应入口：

- `code/vllm/vllm/lora/model_manager.py:285` 到 `model_manager.py:324`
- `code/vllm/vllm/lora/model_manager.py:1123` 到 `model_manager.py:1247`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:48` 到 `lora_model_runner_mixin.py:91`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2193` 到 `gpu_model_runner.py:2201`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3845` 到 `gpu_model_runner.py:3867`
- `code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py:217` 到 `compressed_tensors_moe_wna16.py:240`

---

## 3. 请求层限制：LoRARequest 和 enable-lora

### 3.1 `lora_int_id` 必须大于 0

`LoRARequest.__post_init__()` 会检查：

```python
if self.lora_int_id < 1:
    raise ValueError(f"id must be > 0, got {self.lora_int_id}")
```

位置：`code/vllm/vllm/lora/request.py:39` 到 `request.py:41`

执行期 `InputBatch.request_lora_mapping` 中的 `0` 是“无 LoRA”的哨兵值。

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:477` 到 `gpu_input_batch.py:479`

所以：

```text
LoRA adapter id 从 1 开始；
0 不能作为有效 adapter id。
```

### 3.2 `lora_path` 不能为空

`LoRARequest.__post_init__()` 还会检查：

```python
assert self.lora_path, "lora_path cannot be empty"
```

位置：`request.py:43` 到 `request.py:45`

如果要动态加载，至少要能通过 `lora_path` 找到本地目录或 HF / ModelScope repo。

### 3.3 未启用 LoRA 不能传 LoRARequest

V1 输入处理会在请求进入 EngineCore 前调用：

```python
self._validate_lora(lora_request)
```

位置：`code/vllm/vllm/v1/engine/input_processor.py:256` 到 `input_processor.py:257`

如果请求带了 LoRA，但 `self.lora_config` 不存在：

```python
raise ValueError(f"Got lora_request {lora_request} but LoRA is not enabled!")
```

位置：`input_processor.py:146` 到 `input_processor.py:154`

所以这类错误应该优先检查：

```text
服务启动参数是否启用了 LoRA；
VllmConfig.lora_config 是否为 None；
请求是否意外带了 lora_request。
```

### 3.4 LoRA tokenizer 支持已 deprecated

如果输入处理器持有 tokenizer，并且请求带 LoRA，会打一次 warning：

```text
vLLM has deprecated support for supporting different tokenizers for different LoRAs.
```

位置：`input_processor.py:156` 到 `input_processor.py:163`

这不是失败，但调试输出差异时要注意：默认使用 base model tokenizer。

---

## 4. LoRAConfig 限制

`LoRAConfig` 定义在：`code/vllm/vllm/config/lora.py:30`

### 4.1 rank 限制

配置字段：

```python
max_lora_rank: MaxLoRARanks = 16
```

位置：`lora.py:34` 到 `lora.py:35`

PEFT helper 校验：

```python
if self.r > lora_config.max_lora_rank:
    error_msg.append(
        f"LoRA rank {self.r} is greater than max_lora_rank {lora_config.max_lora_rank}."
    )
```

位置：`code/vllm/vllm/lora/peft_helper.py:120` 到 `peft_helper.py:124`

如果 adapter rank 大于服务配置，加载阶段会失败。

### 4.2 单 batch LoRA 数限制

配置字段：

```python
max_loras: int = Field(default=1, ge=1)
```

位置：`lora.py:36` 到 `lora.py:37`

它有两个含义：

```text
1. Scheduler 单轮 batch 中最多允许多少个不同 LoRA id；
2. LoRAModelManager 中 GPU active slots 的数量。
```

对应位置：

- Scheduler：`scheduler.py:614` 到 `scheduler.py:664`
- manager slots：`code/vllm/vllm/lora/model_manager.py:277` 到 `model_manager.py:283`

### 4.3 CPU cache 容量限制

配置字段：

```python
max_cpu_loras: int | None = None
```

位置：`lora.py:43` 到 `lora.py:45`

校验：

```python
if self.max_cpu_loras is None:
    self.max_cpu_loras = self.max_loras
elif self.max_cpu_loras < self.max_loras:
    raise ValueError(...)
```

位置：`lora.py:108` 到 `lora.py:116`

所以：

```text
max_cpu_loras 必须 >= max_loras；
否则 CPU cache 容量小于 GPU active slots，本身就不一致。
```

### 4.4 不支持 DoRA、bias 和 modules_to_save

PEFT helper 会拒绝：

```text
modules_to_save != None
use_dora=True
bias != "none"
```

位置：

- `peft_helper.py:42` 到 `peft_helper.py:51`
- `peft_helper.py:125` 到 `peft_helper.py:128`

rsLoRA 是支持的，它会把 scaling factor 改成：

```python
lora_alpha / sqrt(r)
```

位置：`peft_helper.py:53` 到 `peft_helper.py:58`

### 4.5 target_modules 会改变可注入模块

`LoRAConfig.target_modules` 用来限制部署时启用 LoRA 的模块。

位置：`lora.py:48` 到 `lora.py:51`

过滤逻辑在：`code/vllm/vllm/lora/utils.py:260` 到 `utils.py:300`

如果 `target_modules=None`，所有支持模块都通过。

如果指定了列表，则要求：

```text
- module suffix 命中 target_modules；或
- 完整 module name 命中 target_modules；或
- packed module parent / child 映射能匹配。
```

常见问题是 adapter 的 PEFT target module 名和 vLLM runtime module 名不同，尤其是 packed qkv、gate/up、MoE experts。

---

## 5. adapter_config.json 和 checkpoint 限制

### 5.1 必须有 PEFT 必要字段

`PEFTHelper` 必要字段是：

```text
r
lora_alpha
target_modules
```

位置：`code/vllm/vllm/lora/peft_helper.py:27` 到 `peft_helper.py:31`

`from_dict()` 会检查缺失字段：

```python
if missing_fields:
    raise ValueError(f"Missing required configuration fields: {missing_fields}")
```

位置：`peft_helper.py:60` 到 `peft_helper.py:78`

### 5.2 adapter_config.json 路径

普通加载时读取：

```text
{lora_path}/adapter_config.json
```

位置：`peft_helper.py:80` 到 `peft_helper.py:112`

如果使用 tensorizer，则从 tensorizer 目录读取。

### 5.3 权重文件格式

`LoRAModel.from_local_checkpoint()` 支持：

```text
adapter_model.safetensors
adapter_model.bin
adapter_model.pt
tensorizer adapter_model.tensors
```

位置：`code/vllm/vllm/lora/lora_model.py:205` 到 `lora_model.py:295`

如果这些都没有：

```python
raise ValueError(f"{lora_dir} doesn't contain tensors")
```

位置：`lora_model.py:294` 到 `lora_model.py:295`

### 5.4 unexpected module 校验

加载时会检查 checkpoint 中的 module 是否在 expected target modules 内。

位置：`lora_model.py:212` 到 `lora_model.py:242`

典型报错：

```text
While loading <lora_dir>, expected target modules in <expected_lora_modules> but received <unexpected_modules>.
Please verify that the loaded LoRA module is correct
```

这通常说明：

```text
- adapter 不是给当前 base model 训练的；
- target_modules 名称和 vLLM 模型模块名不匹配；
- packed module 映射没有覆盖该 checkpoint 命名；
- 使用了错误的 weights mapper。
```

### 5.5 unsupported LoRA weight name

`parse_fine_tuned_lora_name()` 只支持：

```text
...lora_A.weight
...lora_B.weight
...lora_embedding_A
...lora_embedding_B
```

否则抛：

```python
raise ValueError(f"{name} is unsupported LoRA weight")
```

位置：`code/vllm/vllm/lora/utils.py:155` 到 `utils.py:197`

### 5.6 embedding vocab size 必须一致

如果加载 embedding LoRA，且 adapter 的 vocab size 和 base model 不一致，会抛：

```text
The embedding LoRA size(...) must be consistent with the base model's vocabulary size(...).
```

位置：`code/vllm/vllm/lora/lora_model.py:145` 到 `lora_model.py:154`

---

## 6. 模型支持和 target module 调试

### 6.1 base model 必须 supports_lora

V1 加载 LoRA model 时：

```python
if not supports_lora(model):
    raise ValueError(f"{model.__class__.__name__} does not support LoRA yet.")
```

位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:31` 到 `lora_model_runner_mixin.py:39`

这类问题优先查：

```text
- 模型类是否实现 vLLM 的 SupportsLoRA 能力；
- 是否用了不支持 LoRA 的模型 architecture；
- 是否是 pooling / multimodal / MoE 的特殊模型路径。
```

### 6.2 supported modules 如何发现

`get_supported_lora_modules()` 会扫描模型：

```text
- module.embedding_modules 中声明的 embedding modules；
- LinearBase 子类；
- MoERunner 子类。
```

位置：`code/vllm/vllm/lora/utils.py:208` 到 `utils.py:229`

`LoRAModelManager.__init__()` 如果找不到支持模块会 assert：

```python
assert self.supported_lora_modules, (
    f"No supported LoRA modules found in {self.model.__class__.__name__}."
)
```

位置：`code/vllm/vllm/lora/model_manager.py:88` 到 `model_manager.py:92`

### 6.3 wrapper 替换失败

`_create_lora_modules()` 中，命中目标模块后会调用：

```python
from_layer(...)
```

位置：`model_manager.py:441` 到 `model_manager.py:450`

`from_layer()` 会遍历所有 LoRA wrapper class，找到能替换当前 layer 的实现。

位置：`code/vllm/vllm/lora/utils.py:76` 到 `utils.py:124`

如果匹配了 target module，但最终没有得到 `BaseLayerWithLoRA`：

```text
LoRA target module <module_name> (<type>) matched the deployment configuration but could not be wrapped by any LoRA layer implementation.
```

如果用户显式配置了 `target_modules`，会 raise `ValueError`；否则只是 warning 并忽略。

位置：`model_manager.py:481` 到 `model_manager.py:497`

### 6.4 MoE expert mapping 限制

MoE 模型会调用：

```python
process_packed_modules_mapping(model)
```

位置：`model_manager.py:125` 到 `model_manager.py:127`

如果是 MoE 模型但没有 expert mapping：

```python
raise AttributeError(
    "To support LoRA for MoE model, 'get_expert_mapping' must be implemented"
)
```

位置：`code/vllm/vllm/lora/utils.py:360` 到 `utils.py:390`

所以 MoE LoRA 出错时，要查：

```text
- base model 是否提供 get_expert_mapping；
- adapter 是 2D 还是 3D MoE 格式；
- enable_mixed_moe_lora_format 是否符合预期；
- expert parallel 下本 rank local expert slicing 是否走到正确路径。
```

---

## 7. Scheduler 和 max_loras 限制

Scheduler 会在调度阶段约束本轮 batch 的 LoRA 数量。

running 请求收集已调度 LoRA：

```python
scheduled_loras = set(
    req.lora_request.lora_int_id
    for req in scheduled_running_reqs
    if req.lora_request and req.lora_request.lora_int_id > 0
)
assert len(scheduled_loras) <= self.lora_config.max_loras
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:614` 到 `scheduler.py:622`

waiting 请求如果会引入第 `max_loras + 1` 个不同 LoRA，会被跳过：

```python
if len(scheduled_loras) == self.lora_config.max_loras \
   and request.lora_request.lora_int_id not in scheduled_loras:
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

位置：`scheduler.py:651` 到 `scheduler.py:664`

这类问题的现象通常是：

```text
请求不是报错，而是排队等待；
吞吐下降或某些 LoRA 请求迟迟不被调度；
同一批里 LoRA 种类太多时更明显。
```

调试入口：

```text
- 检查 lora_config.max_loras；
- 检查当前 running / waiting 请求的 lora_int_id 分布；
- 检查是否给同一个 adapter 分配了多个不同 id；
- 检查 scheduler 是否因为 max_loras 把请求放回 skipped_waiting。
```

---

## 8. manager cache 和 active slot 限制

### 8.1 CPU registered cache 满

普通 `LoRAModelManager.add_adapter()` 如果 registered 数量达到 capacity，会抛：

```python
raise RuntimeError("No free adapter slots.")
```

位置：`code/vllm/vllm/lora/model_manager.py:1130` 到 `model_manager.py:1137`

V1 默认使用 LRU 版本，新增 adapter 前如果超过 `max_cpu_loras` 会先移除 oldest：

位置：`code/vllm/vllm/lora/worker_manager.py:293` 到 `worker_manager.py:299`

### 8.2 GPU active slot 满

父类 `activate_adapter()` 找不到空 slot 会抛：

```python
raise ValueError("No free lora slots")
```

位置：`model_manager.py:292` 到 `model_manager.py:302`

LRU 版本会在 active adapter 数达到 `lora_slots` 时先 remove oldest：

位置：`model_manager.py:1208` 到 `model_manager.py:1220`

如果太多 adapter 被 pin，LRU 可能无法移除 oldest。

LRU cache 在所有条目都 pinned 时会抛：

```text
All items are pinned, cannot remove oldest from the cache.
```

位置：`code/vllm/vllm/utils/cache.py:189` 到 `cache.py:204`

### 8.3 pin 失败

`pin_lora()` 要求 adapter 已经在 CPU registered cache 中。

如果没注册：

```text
Pinning failed. LoRA <id> is not registered.
```

位置：`code/vllm/vllm/lora/model_manager.py:1234` 到 `model_manager.py:1240`

所以 pin 前要先 add / load adapter。

---

## 9. InputBatch 和 LoRAMapping 调试

### 9.1 request_lora_mapping

`InputBatch` 初始化：

```python
self.request_lora_mapping = np.zeros((self.max_num_reqs,), dtype=np.int64)
self.lora_id_to_request_ids = {}
self.lora_id_to_lora_request = {}
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:244` 到 `gpu_input_batch.py:247`

新增请求时：

```text
有 LoRA：request_lora_mapping[req_index] = lora_id
无 LoRA：request_lora_mapping[req_index] = 0
```

位置：`gpu_input_batch.py:468` 到 `gpu_input_batch.py:479`

### 9.2 移除请求时只清 batch 引用

`remove_request()` 会从 batch 的 LoRA 引用表里删除 request。

位置：`gpu_input_batch.py:530` 到 `gpu_input_batch.py:538`

但它不会从 manager cache 删除 adapter。

```text
如果 list_loras 里还能看到这个 adapter，这是正常的：
adapter 仍在 CPU cache 中等待复用。
```

### 9.3 make_lora_inputs

执行前：

```python
prompt_lora_mapping, token_lora_mapping, active_lora_requests = \
    input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)
```

位置：`gpu_input_batch.py:976` 到 `gpu_input_batch.py:999`

它返回：

```text
prompt_lora_mapping:
  sampled token 维度的 LoRA id 映射。

token_lora_mapping:
  scheduled input token 维度的 LoRA id 映射。

active_lora_requests:
  当前 batch 中需要的 LoRARequest 集合。
```

### 9.4 设置 active LoRA

`GPUModelRunner` 在输入准备阶段调用：

```python
self.set_active_loras(self.input_batch, num_scheduled_tokens, num_sampled_tokens)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2193` 到 `gpu_model_runner.py:2201`

`LoRAModelRunnerMixin._set_active_loras()` 会构造：

```python
LoRAMapping(token_lora_mapping, prompt_lora_mapping, is_prefill=True, type=mapping_type)
```

然后调用：

```python
self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
```

位置：`code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:48` 到 `lora_model_runner_mixin.py:91`

如果输出看起来像 adapter 没生效，要重点检查：

```text
- request_lora_mapping 是否有正确 lora_id；
- lora_id_to_lora_request 是否包含该 id；
- make_lora_inputs 的 token_lora_mapping 是否覆盖了本轮 scheduled tokens；
- LoRAModelManager.lora_index_to_id 是否包含该 id；
- Punica wrapper metadata 是否被 set_adapter_mapping 更新。
```

---

## 10. Multimodal LoRA 特殊限制

### 10.1 tower / connector LoRA 是实验特性

配置字段：

```python
enable_tower_connector_lora: bool = False
```

位置：`code/vllm/vllm/config/lora.py:62` 到 `lora.py:66`

`LoRAModelManager._maybe_init_mm()` 会先判断模型是否支持：

```text
supports_multimodal(model)
hasattr(model, "get_mm_mapping")
hasattr(model, "get_num_mm_encoder_tokens")
```

位置：`code/vllm/vllm/lora/model_manager.py:139` 到 `model_manager.py:269`

如果配置没开但模型支持，会 info 提示需要设置 `enable_tower_connector_lora=True`。

如果配置开了但模型不支持，会 warning 并忽略。

位置：`model_manager.py:188` 到 `model_manager.py:213`

### 10.2 language_model_only 会禁用 tower connector LoRA

如果 multimodal model 配置了 `language_model_only`，会 warning 并关闭 tower connector LoRA。

位置：`model_manager.py:215` 到 `model_manager.py:225`

### 10.3 LoRA 会影响 multimodal cache key

当 `enable_tower_connector_lora=True` 时，同一个图片在不同 LoRA 下可能产生不同 encoder embedding。

因此 input processor 会把 LoRA name 加入 mm identifier：

```python
return f"{lora_request.lora_name}:{mm_hash}"
```

位置：`code/vllm/vllm/v1/engine/input_processor.py:165` 到 `input_processor.py:181`

如果调试 multimodal LoRA cache 命中异常，要检查：

```text
- 是否开启 enable_tower_connector_lora；
- mm_feature.identifier 是否包含 lora_name；
- 同一 LoRA name 是否稳定；
- 不同 adapter 是否错误复用了同一个 lora_name。
```

### 10.4 tower / connector mapping 与 language mapping 不同

multimodal encoder 执行时会单独设置 tower / connector LoRA mapping。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2941` 到 `gpu_model_runner.py:3008`

这类问题要区分：

```text
language LoRA 生效，不代表 vision tower LoRA 生效；
tower LoRA 生效，不代表 connector LoRA 生效；
三者使用不同 LoRAMappingType 和 token 数。
```

---

## 11. MoE、并行和量化组合限制

### 11.1 MoE LoRA 依赖 packed/expert mapping

MoE 会走 `FusedMoEWithLoRA` 或 `FusedMoE3DWithLoRA`。

相关 wrapper 在：`code/vllm/vllm/lora/utils.py:76` 到 `utils.py:95`

如果 MoE 模型无法提供 expert mapping，会抛：

```text
To support LoRA for MoE model, 'get_expert_mapping' must be implemented
```

位置：`utils.py:360` 到 `utils.py:390`

### 11.2 Expert Parallel 下会裁剪 local experts

LoRA manager 会构造 `MoEEPLoadSpec`，让 loader 在读 checkpoint 时跳过非本 rank 的 experts。

位置：

- `code/vllm/vllm/lora/model_manager.py:1084` 到 `model_manager.py:1105`
- `code/vllm/vllm/lora/lora_model.py:25` 到 `lora_model.py:58`
- `lora_model.py:271` 到 `lora_model.py:293`

如果 EP 下 LoRA 输出异常，要查：

```text
- ep_rank / local_num_experts / global_num_experts 是否正确；
- checkpoint expert index 命名是否符合 .experts.<idx>. 格式；
- 2D / 3D MoE LoRA 格式是否和配置匹配。
```

### 11.3 mixed MoE LoRA format

配置：

```python
enable_mixed_moe_lora_format: bool = False
```

位置：`code/vllm/vllm/config/lora.py:74` 到 `lora.py:79`

开启后，manager 会强制使用 universal 2D wrapper，以便同一部署中混用 2D / 3D MoE adapter。

相关逻辑：

- `code/vllm/vllm/lora/model_manager.py:114` 到 `model_manager.py:128`
- `model_manager.py:912` 到 `model_manager.py:1003`

### 11.4 量化 backend 不是全部都等价支持 LoRA

不要简单理解成“量化都支持 LoRA”或“量化都不支持 LoRA”。限制通常来自具体 backend。

例子：compressed tensors MoE WNA16 在 LoRA 开启时需要 Triton experts：

```python
if self.moe.is_lora_enabled:
    if HAS_TRITON:
        return TritonWNA16Experts(...)
    else:
        raise NotImplementedError(
            "TritonExperts requires Triton. Install triton or disable LoRA for MoE."
        )
```

位置：`code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py:217` 到 `compressed_tensors_moe_wna16.py:240`

调试量化 + LoRA 时，要定位到：

```text
- base layer 的 quant_method；
- LoRA wrapper 是否正确保留 base_layer.quant_method；
- MoE backend 是否支持 LoRA；
- 是否需要 Triton；
- 当前平台 CUDA / ROCm / CPU 是否有对应 kernel。
```

---

## 12. CUDA graph 和 LoRA 调试

LoRA 会影响 CUDA graph dispatch，因为有无 LoRA、active LoRA 数不同，会改变执行路径和 kernel metadata。

### 12.1 `cudagraph_specialize_lora`

配置在：`code/vllm/vllm/config/compilation.py:641`

含义：

```text
True：分别捕获有 LoRA / 无 LoRA 的图，降低无 LoRA 请求的额外开销；
False：所有情况都使用 LoRA-enabled graph，减少图数量，但无 LoRA 时也有 LoRA op 开销。
```

位置：`compilation.py:641` 到 `compilation.py:648`

### 12.2 `specialize_active_lora`

配置在：`code/vllm/vllm/config/lora.py:67`

含义：

```text
按 active LoRA adapter 数量进一步构造 LoRA kernel grid / capture key；
可能提升不同 LoRA 使用模式下的性能；
代价是 startup time 和显存占用增加。
```

位置：`lora.py:67` 到 `lora.py:73`

### 12.3 capture active LoRA 数

`GPUModelRunner` dispatch CUDA graph 前会计算：

```python
num_active_loras = len(self.input_batch.lora_id_to_lora_request)
has_lora = num_active_loras > 0
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3845` 到 `gpu_model_runner.py:3867`

LoRA capture 数量 helper：

```python
get_captured_lora_counts(max_loras, specialize)
```

位置：`code/vllm/vllm/lora/utils.py:49` 到 `utils.py:64`

调试建议：

```text
- 如果问题只在 CUDA graph 开启时出现，先强制 eager / 禁用对应 graph 模式复现；
- 检查 num_active_loras 和实际 batch 中 LoRA id 数是否一致；
- 检查 cudagraph_specialize_lora / specialize_active_lora 是否改变 capture key；
- 检查 warmup dummy LoRA 是否覆盖了当前 active LoRA 数量。
```

---

## 13. 常见报错按文本定位

### 13.1 `Got lora_request ... but LoRA is not enabled!`

位置：`code/vllm/vllm/v1/engine/input_processor.py:150` 到 `input_processor.py:154`

含义：请求传了 LoRA，但服务没有启用 LoRA。

检查：

```text
- 启动参数 / VllmConfig.lora_config；
- 调用方是否错误传入 lora_request；
- 是否某个默认 multimodal LoRA 注入逻辑意外生效。
```

### 13.2 `id must be > 0`

位置：`code/vllm/vllm/lora/request.py:39` 到 `request.py:41`

含义：`lora_int_id` 非法。

检查：

```text
- adapter id 生成逻辑；
- 是否把 0 当成 adapter id；
- 同一 adapter 在多 engine 中 id 是否稳定。
```

### 13.3 `lora_path cannot be empty`

位置：`request.py:43` 到 `request.py:45`

含义：`LoRARequest.lora_path` 为空。

检查：

```text
- API 层模型名到 LoRA path 的解析；
- 动态加载请求体；
- LoRA resolver 是否返回了路径。
```

### 13.4 `LoRA rank ... is greater than max_lora_rank ...`

位置：`code/vllm/vllm/lora/peft_helper.py:120` 到 `peft_helper.py:124`

含义：adapter rank 超过服务配置。

处理：

```text
- 提高 max_lora_rank；或
- 使用 rank 更小的 adapter；
- 确认 adapter_config.json 中 r 是否符合预期。
```

### 13.5 `vLLM does not yet support DoRA` / `Adapter bias is not supported`

位置：`peft_helper.py:42` 到 `peft_helper.py:51`，`peft_helper.py:125` 到 `peft_helper.py:128`

含义：adapter 使用了当前 vLLM 不支持的 PEFT 特性。

处理：换普通 LoRA adapter，或重新训练/导出不带 DoRA、bias、modules_to_save 的版本。

### 13.6 `... doesn't contain tensors`

位置：`code/vllm/vllm/lora/lora_model.py:294` 到 `lora_model.py:295`

含义：adapter 目录没有支持的权重文件。

检查：

```text
adapter_model.safetensors
adapter_model.bin
adapter_model.pt
tensorizer adapter_model.tensors
```

### 13.7 `expected target modules ... but received ...`

位置：`lora_model.py:236` 到 `lora_model.py:242`

含义：checkpoint 中的 LoRA module 名和当前模型期望不匹配。

检查：

```text
- base model 是否和 adapter 匹配；
- adapter target_modules；
- vLLM target_modules 限制；
- packed module mapping；
- hf_to_vllm_mapper；
- pool model 是否有 model. 前缀差异。
```

### 13.8 `unsupported LoRA weight`

位置：`code/vllm/vllm/lora/utils.py:155` 到 `utils.py:197`

含义：checkpoint 权重名不是 vLLM 支持的 LoRA A/B 命名。

检查 adapter 文件 key。

### 13.9 `No free adapter slots.`

位置：`code/vllm/vllm/lora/model_manager.py:1130` 到 `model_manager.py:1137`

含义：非 LRU manager 的 registered adapter 容量满，或绕过 LRU 行为。

检查 `max_cpu_loras` 和 manager 类型。

### 13.10 `No free lora slots`

位置：`model_manager.py:292` 到 `model_manager.py:302`

含义：没有可用 GPU active slot。

检查：

```text
- max_loras；
- active adapters；
- 是否 pin 了太多 adapter；
- 是否使用 LRU manager。
```

### 13.11 `Pinning failed. LoRA ... is not registered.`

位置：`model_manager.py:1234` 到 `model_manager.py:1240`

含义：pin 前没有 add / load 该 adapter。

### 13.12 `All items are pinned, cannot remove oldest from the cache.`

位置：`code/vllm/vllm/utils/cache.py:189` 到 `cache.py:204`

含义：LRU 需要淘汰，但所有候选都被 pin。

处理：取消部分 pin、增大容量，或减少同时使用的 adapter。

### 13.13 `TritonExperts requires Triton. Install triton or disable LoRA for MoE.`

位置：`code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py:222` 到 `compressed_tensors_moe_wna16.py:238`

含义：当前 MoE quant backend + LoRA 路径需要 Triton。

---

## 14. 调试入口清单

按定位顺序看这些点：

```text
1. 请求层
   - LoRARequest.lora_name / lora_int_id / lora_path
   - lora_int_id 是否 > 0
   - lora_path 是否存在或可下载

2. 配置层
   - VllmConfig.lora_config 是否为 None
   - max_lora_rank / max_loras / max_cpu_loras
   - target_modules
   - enable_tower_connector_lora
   - specialize_active_lora

3. adapter 文件
   - adapter_config.json
   - adapter_model.safetensors / bin / pt
   - adapter_config.json 中 r / bias / use_dora / modules_to_save / target_modules
   - safetensors keys 是否是 lora_A / lora_B 命名

4. 模型注入
   - supports_lora(model)
   - get_supported_lora_modules(model)
   - LoRAModelManager.modules
   - LoRA wrapper 是否替换了目标模块
   - packed_modules_mapping

5. manager 状态
   - list_loras()
   - _registered_adapters
   - _active_adapters
   - lora_index_to_id
   - pinned items

6. 调度状态
   - scheduled_loras
   - waiting / skipped_waiting
   - 是否超过 max_loras

7. batch 状态
   - InputBatch.request_lora_mapping
   - lora_id_to_request_ids
   - lora_id_to_lora_request
   - make_lora_inputs() 输出的 token_lora_mapping / prompt_lora_mapping

8. forward 状态
   - set_active_loras() 是否被调用
   - LoRAMapping.type 是否正确
   - Punica wrapper metadata 是否更新
   - CUDA graph dispatch 的 has_lora / num_active_loras
   - quant / MoE backend 是否支持当前组合
```

---

## 15. 推荐排查流程

### 15.1 请求一发出就报错

优先查：

```text
LoRARequest.__post_init__()
InputProcessor._validate_lora()
LoRAConfig._validate_lora_config()
PEFTHelper.validate_legal()
```

对应源码：

- `request.py:39` 到 `request.py:45`
- `input_processor.py:146` 到 `input_processor.py:154`
- `lora.py:108` 到 `lora.py:125`
- `peft_helper.py:114` 到 `peft_helper.py:128`

### 15.2 adapter 加载失败

优先查：

```text
WorkerLoRAManager._load_adapter()
PEFTHelper.from_local_dir()
LoRAModel.from_local_checkpoint()
parse_fine_tuned_lora_name()
```

对应源码：

- `worker_manager.py:99` 到 `worker_manager.py:162`
- `peft_helper.py:80` 到 `peft_helper.py:112`
- `lora_model.py:167` 到 `lora_model.py:306`
- `utils.py:155` 到 `utils.py:197`

### 15.3 请求不报错但迟迟不执行

优先查 Scheduler：

```text
- 本轮 scheduled_loras 数；
- request.lora_request.lora_int_id 是否引入新 LoRA；
- len(scheduled_loras) 是否已经等于 max_loras；
- 请求是否被放入 skipped_waiting。
```

对应源码：`scheduler.py:651` 到 `scheduler.py:664`

### 15.4 请求执行但 LoRA 不生效

沿执行链看：

```text
InputBatch.add_request()
  → request_lora_mapping 是否是目标 lora_id
make_lora_inputs()
  → token_lora_mapping 是否覆盖本轮 tokens
set_active_loras()
  → active_lora_requests 是否包含目标 LoRARequest
LoRAModelManager.activate_adapter()
  → lora_index_to_id 是否包含目标 id
set_adapter_mapping()
  → Punica wrapper metadata 是否更新
```

对应源码：

- `gpu_input_batch.py:468` 到 `gpu_input_batch.py:479`
- `gpu_input_batch.py:976` 到 `gpu_input_batch.py:999`
- `lora_model_runner_mixin.py:48` 到 `lora_model_runner_mixin.py:91`
- `model_manager.py:285` 到 `model_manager.py:324`
- `model_manager.py:344` 到 `model_manager.py:367`

### 15.5 只有 multimodal LoRA 不生效

优先查：

```text
- enable_tower_connector_lora 是否开启；
- model 是否 supports_mm 且有 get_mm_mapping；
- model 是否有 get_num_mm_encoder_tokens；
- mm_feature.identifier 是否包含 lora_name；
- _execute_mm_encoder() 中 tower / connector mapping 是否设置；
- LoRAMappingType 是 TOWER / CONNECTOR 还是 LANGUAGE。
```

对应源码：

- `model_manager.py:139` 到 `model_manager.py:269`
- `input_processor.py:165` 到 `input_processor.py:181`
- `gpu_model_runner.py:2941` 到 `gpu_model_runner.py:3008`

### 15.6 只有 CUDA graph 开启时异常

优先查：

```text
- cudagraph_specialize_lora；
- specialize_active_lora；
- num_active_loras；
- dummy LoRA warmup 是否覆盖当前 active LoRA 数；
- 切到 eager 后是否消失。
```

对应源码：

- `config/compilation.py:641` 到 `compilation.py:648`
- `config/lora.py:67` 到 `lora.py:73`
- `gpu_model_runner.py:3845` 到 `gpu_model_runner.py:3867`
- `lora_model_runner_mixin.py:93` 到 `lora_model_runner_mixin.py:272`

---

## 16. 最小心智模型

LoRA 调试可以始终按这个图走：

```text
LoRARequest 合法吗？
  → lora_int_id > 0，path 不为空，LoRA 已启用

adapter 文件能加载吗？
  → adapter_config.json，rank，bias/DoRA/modules_to_save，权重文件，module key

base model 能注入吗？
  → supports_lora，supported modules，target_modules，packed mapping，MoE mapping

这一轮能调度吗？
  → 不同 LoRA 数 <= max_loras

adapter 在 manager 里吗？
  → list_loras，registered cache，active slots，pin / LRU

mapping 正确吗？
  → request_lora_mapping，token_lora_mapping，prompt_lora_mapping，lora_index_to_id

forward 后端支持吗？
  → Punica wrapper，CUDA graph key，quant backend，MoE / multimodal 特殊路径
```

---

## 17. 一句话总结

LoRA 限制主要来自四类边界：PEFT adapter 本身是否合法，base model 的目标模块能否被 vLLM LoRA wrapper 替换，Scheduler/manager 的 `max_loras` 与 cache/slot 容量是否足够，以及当前 forward 后端是否支持对应的 multimodal、MoE、量化和 CUDA graph 组合；调试时沿 `LoRARequest → Scheduler → InputBatch → WorkerLoRAManager → LoRAModelManager → LoRAMapping → forward backend` 逐层排查，最快能定位问题发生在哪个契约上。
