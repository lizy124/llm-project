# 10. LoRA 和量化如何共存？

源码位置：

- `code/vllm/vllm/config/lora.py`
- `code/vllm/vllm/engine/arg_utils.py`
- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/lora/model_manager.py`
- `code/vllm/vllm/lora/worker_manager.py`
- `code/vllm/vllm/lora/lora_model.py`
- `code/vllm/vllm/lora/lora_weights.py`
- `code/vllm/vllm/lora/layers/base.py`
- `code/vllm/vllm/lora/layers/base_linear.py`
- `code/vllm/vllm/lora/layers/column_parallel_linear.py`
- `code/vllm/vllm/lora/layers/row_parallel_linear.py`
- `code/vllm/vllm/lora/layers/fused_moe.py`
- `code/vllm/vllm/lora/punica_wrapper/`
- `code/vllm/vllm/lora/ops/triton_ops/`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/layers/fused_moe/experts/lora_context.py`
- `code/vllm/vllm/model_executor/layers/fused_moe/experts/lora_experts_mixin.py`
- `code/vllm/vllm/model_executor/layers/fused_moe/experts/triton_moe.py`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：当 base model 使用 AWQ / GPTQ / FP8 / compressed-tensors / online quant 等量化方式时，LoRA adapter 如何加载、如何挂到量化 layer 上、forward 时如何把 LoRA delta 与量化 base output 合并；以及 Linear、MoE、TP、EP、CUDA graph、LoRA dtype、adapter hot-swap 等组合有哪些限制。

---

## 1. 一句话回答

LoRA 和量化在 vLLM 中不是把 LoRA adapter 直接合并进低 bit base 权重，而是：

```text
base model 按 quant_config 创建量化 layer 和 quant_method；
LoRA manager 再把可支持的 Linear / MoE / Embedding / logits layer 包成 LoRA wrapper；
forward 时 wrapper 先调用 base_layer.quant_method.apply() 得到量化 base output；
再用 Punica kernel 计算 LoRA A/B 的低秩增量；
最后把 LoRA delta 加到 base output 上。
```

普通 Linear 主链路是：

```text
Quantized LinearBase
  → LoRA wrapper 保存 base_layer
  → wrapper.apply()
  → base_layer.quant_method.apply(base_layer, x, bias)
  → punica_wrapper.add_lora_linear(output, x, lora_a, lora_b)
  → output = quantized_base_output + lora_delta
```

所以：

```text
base 权重可以是低 bit / FP8 / packed layout；
LoRA adapter 权重通常是 float16 / bfloat16，单独存放在 LoRA slots；
两者在 forward 输出空间相加，而不是在权重存储层面合并。
```

---

## 2. 最小主链路

从请求到量化 layer + LoRA 生效，可以压缩成：

```text
EngineArgs(enable_lora=True, quantization=...)
  → VllmConfig.lora_config + VllmConfig.quant_config
  → load quantized base model
  → LoRAModelRunnerMixin.load_lora_model()
  → WorkerLoRAManager.create_lora_manager()
  → LoRAModelManager._create_lora_modules()
  → replace LinearBase / MoERunner with BaseLayerWithLoRA wrapper
  → request 携带 LoRARequest
  → InputBatch.make_lora_inputs()
  → LoRAModelRunnerMixin.set_active_loras()
  → WorkerLoRAManager.set_active_adapters()
  → LoRA adapter 权重 copy 到 GPU stacked buffers
  → forward: quantized base output + LoRA delta
```

这里有三类对象要分清：

```text
QuantizationConfig / quant_method：
  决定 base layer 权重如何存、如何算。

LoRAConfig / LoRAModel / LoRALayerWeights：
  决定 adapter 数量、rank、dtype、加载路径、A/B 权重。

PunicaWrapper / LoRAKernelMeta：
  决定当前 batch 每个 token 用哪个 LoRA slot，以及如何批量执行 LoRA delta。
```

---

## 3. 配置入口：LoRA 和 quantization 是两条配置线

### 3.1 EngineArgs 中的 LoRA 参数

LoRA 参数在 `engine/arg_utils.py` 中定义。

关键字段：

```text
enable_lora
max_loras
max_lora_rank
default_mm_loras
fully_sharded_loras
max_cpu_loras
lora_dtype
lora_target_modules
enable_tower_connector_lora
specialize_active_lora
enable_mixed_moe_lora_format
```

位置：`engine/arg_utils.py:586` 到 `engine/arg_utils.py:596`

CLI 参数入口在：

```text
--enable-lora
--max-loras
--max-lora-rank
--lora-dtype
--max-cpu-loras
--fully-sharded-loras
--lora-target-modules
```

位置：`engine/arg_utils.py:1297` 到 `engine/arg_utils.py:1324`

### 3.2 EngineArgs 创建 LoRAConfig

只有 `enable_lora=True` 时才会创建：

```python
lora_config = LoRAConfig(...) if self.enable_lora else None
```

位置：`engine/arg_utils.py:2145` 到 `engine/arg_utils.py:2161`

随后放进：

```python
VllmConfig(lora_config=lora_config, ...)
```

位置：`engine/arg_utils.py:2327`

### 3.3 VllmConfig 同时持有 lora_config 和 quant_config

`VllmConfig` 中有两个独立字段：

```python
lora_config: LoRAConfig | None = None
quant_config: QuantizationConfig | None = None
```

位置：`config/vllm.py:319`、`config/vllm.py:334`

这说明：

```text
LoRA 是否启用，与 base model 是否量化是两套开关。
```

常见组合：

```text
不量化 + LoRA
量化 base + 不启用 LoRA
量化 base + LoRA
只量化 KV cache + LoRA
```

---

## 4. LoRAConfig 如何校验 dtype / rank / slots

`LoRAConfig` 在：`config/lora.py:30`

关键字段：

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

位置：`config/lora.py:34` 到 `config/lora.py:79`

### 4.1 rank 不是任意整数

`max_lora_rank` 类型是：

```python
Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]
```

位置：`config/lora.py:25` 到 `config/lora.py:27`

所以部署时 LoRA rank 要落在这些候选值里。

### 4.2 lora_dtype 默认跟随 base model dtype

`VllmConfig.__post_init__()` 会调用：

```python
self.lora_config.verify_with_model_config(self.model_config)
```

位置：`config/vllm.py:895` 到 `config/vllm.py:896`

`verify_with_model_config()` 的逻辑是：

```python
if self.lora_dtype in (None, "auto"):
    self.lora_dtype = model_config.dtype
elif isinstance(self.lora_dtype, str):
    self.lora_dtype = getattr(torch, self.lora_dtype)
```

位置：`config/lora.py:127` 到 `config/lora.py:131`

这很重要：

```text
base 权重即使是 4bit / 8bit / FP8，model_config.dtype 通常仍是 float16 / bfloat16；
LoRA adapter buffer 默认用这个 activation dtype，而不是 base 权重的低 bit dtype。
```

### 4.3 max_cpu_loras 必须覆盖 max_loras

`LoRAConfig._validate_lora_config()` 会保证：

```text
max_cpu_loras >= max_loras
```

位置：`config/lora.py:108` 到 `config/lora.py:116`

含义是：

```text
GPU 上最多同时激活 max_loras 个 adapter；
CPU cache 至少要能容纳这些 adapter。
```

### 4.4 dual stream 和 fully sharded 有冲突

如果开启环境变量 `VLLM_LORA_ENABLE_DUAL_STREAM`，但又启用 `fully_sharded_loras`，会警告并关闭 dual stream。

位置：`config/lora.py:117` 到 `config/lora.py:124`

这会影响 LoRA 和量化 base GEMM 是否能并行重叠。

---

## 5. base model 量化先发生，LoRA wrapper 后挂上去

模型加载阶段，base model 先根据 `VllmConfig.quant_config` 创建量化 layer。

普通 Linear 在 `linear.py` 中会先确定：

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
```

位置：`model_executor/layers/linear.py:268` 到 `linear.py:274`

这一步之后，base layer 已经知道：

```text
用 unquantized GEMM、AWQ、GPTQ、FP8、compressed-tensors、online quant，还是别的 quant method。
```

如果启用 LoRA，模型 runner 再调用：

```python
LoRAModelRunnerMixin.load_lora_model(model, vllm_config, device)
```

位置：`v1/worker/lora_model_runner_mixin.py:31` 到 `lora_model_runner_mixin.py:47`

这里会创建：

```python
LRUCacheWorkerLoRAManager(...)
```

并执行：

```python
return self.lora_manager.create_lora_manager(model, vllm_config)
```

所以顺序可以理解成：

```text
先得到一个可运行的 quantized base model；
再把其中支持 LoRA 的 module 替换成 LoRA wrapper；
wrapper 内部保留原始 base_layer，不破坏 base_layer.quant_method。
```

---

## 6. LoRA manager 如何找到要替换的 layer

`LoRAModelManager.__init__()` 会做几件事：

```text
1. 找模型支持哪些 LoRA module；
2. 处理 packed_modules_mapping；
3. 初始化 PunicaWrapper；
4. 调用 _create_lora_modules() 替换 module。
```

位置：`lora/model_manager.py:95` 到 `lora/model_manager.py:140`

### 6.1 支持哪些 module

`get_supported_lora_modules()` 中写得很直接：

```python
if isinstance(module, (LinearBase,)):
    supported_lora_modules.add(name.split(".")[-1])

if isinstance(module, (MoERunner,)):
    supported_lora_modules.add(name.split(".")[-1])
```

位置：`lora/utils.py:219` 到 `lora/utils.py:240`

也就是说：

```text
vLLM 认为所有 LinearBase 都可以尝试挂 LoRA；
MoERunner 也可以尝试挂 MoE LoRA。
```

这里的 `LinearBase` 包括量化 Linear，因为量化是通过 `base_layer.quant_method` 实现的，并没有改变 layer 仍是 `LinearBase` 这个事实。

### 6.2 target_modules 进一步过滤

`LoRAConfig.target_modules` 可以限制部署时允许哪些 module 使用 LoRA。

判断在：

```python
is_in_target_modules(module_name, target_modules, packed_modules_mapping)
```

位置：`lora/utils.py:271` 到 `lora/utils.py:311`

它支持两种匹配：

```text
直接匹配 runtime module suffix；
通过 packed_modules_mapping 匹配打包层里的逻辑子层。
```

这对量化模型很重要，因为 QKV、gate/up 等层经常是 packed module。

---

## 7. packed module 如何对齐 LoRA 和量化层

量化模型里常见 packed layer：

```text
q_proj + k_proj + v_proj → qkv_proj
up_proj + gate_proj → gate_up_proj
多个专家权重 → experts
```

LoRA adapter 的权重名通常还是面向原始 HF module：

```text
q_proj.lora_A / q_proj.lora_B
k_proj.lora_A / k_proj.lora_B
v_proj.lora_A / v_proj.lora_B
```

但 vLLM runtime 里可能只有一个 packed layer。

因此 LoRA manager 会建立：

```text
packed_modules_mapping
```

位置：`lora/model_manager.py:131` 到 `model_manager.py:133`

生成逻辑在：

```python
process_packed_modules_mapping(model, force_2d_moe=...)
```

位置：`lora/utils.py:371` 到 `lora/utils.py:403`

替换 layer 时会取：

```python
parts = module_name.split(".")[-1]
packed_moduled_lst = self.packed_modules_mapping.get(parts, [])
```

位置：`lora/model_manager.py:439` 到 `model_manager.py:447`

加载 adapter 后，又会把多个子 LoRA 合并成一个 packed LoRA：

```python
PackedLoRALayerWeights.pack(...)
PackedLoRALayerWeights.pack_moe(...)
```

位置：`lora/model_manager.py:731` 到 `model_manager.py:782`

这说明：

```text
量化 packed 权重和 LoRA adapter 的逻辑子模块名不同；
vLLM 靠 packed_modules_mapping 把 adapter 的子层 A/B 权重重新打包，
再复制到 runtime packed layer 的 LoRA buffer 中。
```

---

## 8. LoRA wrapper 如何替换 base layer

核心入口：

```python
from_layer(layer, max_loras, lora_config, packed_modules_list, model_config)
```

位置：`lora/utils.py:106` 到 `lora/utils.py:124`

它会按顺序尝试这些 wrapper：

```text
VocabParallelEmbeddingWithLoRA
ColumnParallelLinearWithLoRA
MergedColumnParallelLinearWithLoRA
QKVParallelLinearWithLoRA
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

一旦某个 wrapper 的：

```python
can_replace_layer(source_layer, lora_config, packed_modules_list, model_config)
```

返回 True，就会：

```python
instance_layer = lora_cls(layer)
instance_layer.create_lora_weights(max_loras, lora_config, model_config)
return instance_layer
```

位置：`lora/utils.py:113` 到 `lora/utils.py:123`

替换发生在：

```python
replace_submodule(self.model, module_name, from_layer(...))
```

位置：`lora/model_manager.py:448` 到 `model_manager.py:458`

替换后，LoRA manager 还会把同一个 Punica wrapper 绑定到 wrapper 上：

```python
new_module.set_mapping(punica_wrapper)
```

位置：`lora/model_manager.py:506` 到 `model_manager.py:508`

---

## 9. 普通 Linear LoRA wrapper 如何保留量化 base

`BaseLinearLayerWithLoRA` 的构造函数保存：

```python
self.base_layer = base_layer
self.input_size = self.base_layer.input_size
self.tp_size = self.base_layer.tp_size
self.tp_rank = self.base_layer.tp_rank
```

位置：`lora/layers/base_linear.py:70` 到 `base_linear.py:81`

注意：

```text
它没有复制 base 权重，也没有反量化 base 权重；
它只是包住原始 base layer。
```

base layer 的量化方法通过：

```python
def _get_quant_method(self) -> QuantizeMethodBase:
    quant_method = self.base_layer.quant_method
```

位置：`lora/layers/base_linear.py:186` 到 `base_linear.py:192`

所以 LoRA wrapper 可以直接复用：

```text
AWQLinearMethod
AutoGPTQLinearMethod
Fp8LinearMethod
CompressedTensorsLinearMethod
UnquantizedLinearMethod
...
```

只要这个 quant method 已经实现了 `apply()`。

---

## 10. 普通 Linear 的 forward 如何合并 LoRA delta

同步路径在：

```python
def _apply_sync(self, x, bias=None):
    output = self._get_quant_method().apply(self.base_layer, x, bias)
    return self._apply_lora_to_output(x, output)
```

位置：`lora/layers/base_linear.py:204` 到 `base_linear.py:208`

LoRA delta 的计算在：

```python
self.punica_wrapper.add_lora_linear(
    output,
    x,
    self.lora_a_stacked,
    self.lora_b_stacked,
    1.0,
    self.output_slices,
)
```

位置：`lora/layers/base_linear.py:215` 到 `base_linear.py:229`

这里的语义是：

```text
output 已经是 base quantized GEMM 的结果；
Punica 在 output 上原地加 LoRA delta；
最终返回 output。
```

等价数学形式是：

```text
y = QuantizedLinear(x; W_base_quantized) + (x @ A_lora @ B_lora) * scale
```

但实现上：

```text
scale 通常已经在 LoRALayerWeights.optimize() 中合进 lora_b；
所以 add_lora_linear() 里传入 scale=1.0。
```

`LoRALayerWeights.optimize()` 位置：`lora/lora_weights.py:36` 到 `lora_weights.py:42`

---

## 11. LoRA A/B 权重是否也量化

通常不是。

`BaseLinearLayerWithLoRA.create_lora_weights()` 创建的是：

```python
torch.zeros(..., dtype=lora_config.lora_dtype, device=self.device)
```

位置：`lora/layers/base_linear.py:129` 到 `base_linear.py:150`

`lora_config.lora_dtype` 来自：

```text
auto → model_config.dtype
或用户指定 float16 / bfloat16
```

位置：`config/lora.py:127` 到 `config/lora.py:131`

加载 adapter 时也会按这个 dtype 转换：

```python
lora_a = tensor.to(device=device, dtype=dtype)
lora_b = tensor.to(device=device, dtype=dtype)
```

位置：`lora/lora_model.py:155` 到 `lora_model.py:162`

所以通常组合是：

```text
base W：4bit / int8 / fp8 / packed low-bit
LoRA A/B：float16 或 bfloat16
activation x：float16 或 bfloat16
output：float16 或 bfloat16
```

不过源码里也存在 FP8 LoRA Triton ops：

```text
lora_shrink_fp8
lora_expand_fp8
fused_moe_lora_fp8
```

位置：`lora/ops/triton_ops/__init__.py:5` 到 `__init__.py:19`

这些 kernel 支持 FP8 输入 / 权重 / scale 参数，但普通 LoRA adapter 加载路径默认并不会把 PEFT adapter 权重量化成 FP8。

---

## 12. PunicaWrapper 在 LoRA + 量化里负责什么

`PunicaWrapperGPU` 的定位是：

```text
维护 Multi-LoRA 运行时 metadata，并给 Triton Punica kernel 提供参数。
```

位置：`lora/punica_wrapper/punica_gpu.py:32` 到 `punica_gpu.py:38`

初始化时会创建两套 `LoRAKernelMeta`：

```text
token_mapping_meta：用于 token-level LoRA linear；
prompt_mapping_meta：用于 logits / sampling 侧 LoRA。
```

位置：`lora/punica_wrapper/punica_gpu.py:57` 到 `punica_gpu.py:73`

### 12.1 add_lora_linear 的语义

```python
buffer = torch.empty((len(output_slices), x.size(0), r), dtype=torch.float32)
self.add_shrink(buffer, x, lora_a_stacked, scale)
self.add_expand(y, buffer, lora_b_stacked, output_slices, add_inputs=True)
```

位置：`lora/punica_wrapper/punica_gpu.py:203` 到 `punica_gpu.py:264`

也就是两段 GEMM：

```text
shrink: x @ A_lora → rank 维 buffer
expand: buffer @ B_lora → hidden/output 维 delta
```

其中 `add_inputs=True` 表示：

```text
把 delta 加到已有 y 上。
```

这里的已有 y 通常就是量化 base layer 的输出。

### 12.2 no-LoRA token 会跳过

`LoRAKernelMeta.prepare_tensors()` 会检查：

```python
no_lora = torch.all(token_lora_mapping == -1)
```

如果全是 no-lora，kernel 早退。

位置：`lora/ops/triton_ops/lora_kernel_metadata.py:118` 到 `lora_kernel_metadata.py:127`

所以同一个 batch 可以混合：

```text
部分 token 使用 adapter A；
部分 token 使用 adapter B；
部分 token 不使用 LoRA。
```

---

## 13. request 级 LoRA 如何变成 token mapping

V1 中，ModelRunner 通过 mixin 设置 active LoRA。

入口：

```python
LoRAModelRunnerMixin.set_active_loras(...)
```

位置：`v1/worker/lora_model_runner_mixin.py:73` 到 `lora_model_runner_mixin.py:91`

它会调用：

```python
input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)
```

得到：

```text
prompt_lora_mapping
token_lora_mapping
lora_requests
```

位置：`v1/worker/lora_model_runner_mixin.py:83` 到 `lora_model_runner_mixin.py:88`

随后构造：

```python
LoRAMapping(token_lora_mapping, prompt_lora_mapping, is_prefill=True, type=...)
```

位置：`v1/worker/lora_model_runner_mixin.py:57` 到 `lora_model_runner_mixin.py:67`

最后：

```python
self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
```

这会做两件事：

```text
1. 确保请求需要的 adapter 已加载并激活；
2. 更新 PunicaWrapper 的 token→LoRA slot metadata。
```

---

## 14. adapter 如何加载到 GPU LoRA slots

Worker 侧管理器是：

```python
LRUCacheWorkerLoRAManager
```

位置：`lora/worker_manager.py:237`

当新请求需要 LoRA 时：

```python
add_adapter(lora_request)
  → _load_adapter(lora_request)
  → _adapter_manager.add_adapter(lora)
  → _adapter_manager.activate_adapter(lora_request.lora_int_id)
```

位置：`lora/worker_manager.py:281` 到 `worker_manager.py:315`

### 14.1 加载 adapter checkpoint

`WorkerLoRAManager._load_adapter()` 会：

```text
1. 解析 adapter 路径；
2. 读取 adapter_config.json / PEFT 配置；
3. 校验 PEFT 配置是否合法；
4. 调用 LoRAModel.from_local_checkpoint() 加载 adapter_model.safetensors / bin / pt；
5. 转成 LoRALayerWeights。
```

位置：`lora/worker_manager.py:105` 到 `worker_manager.py:168`

### 14.2 LoRA 权重先加载到 CPU

这里传入：

```python
device="cpu"
dtype=self.lora_config.lora_dtype
```

位置：`lora/worker_manager.py:138` 到 `worker_manager.py:150`

也就是说 adapter 通常先进入 CPU cache，再按需 copy 到 GPU slot。

### 14.3 激活 adapter 时复制到 stacked buffer

`LoRAModelManager.activate_adapter()` 会找 free slot：

```python
self.lora_index_to_id[index] = lora_model.id
```

然后对每个 LoRA wrapper：

```python
module.set_lora(index, module_lora.lora_a, module_lora.lora_b)
```

位置：`lora/model_manager.py:292` 到 `model_manager.py:330`

`set_lora()` 内部会把 adapter 的 A/B 权重切片后 copy 到：

```text
lora_a_stacked[index]
lora_b_stacked[index]
```

这就是 hot-swap 的基础：

```text
请求改变 active LoRA 时，不重建 base quantized model；
只更新 LoRA slot 和 mapping。
```

---

## 15. ColumnParallel / RowParallel 下如何切 LoRA

LoRA 权重必须和 base layer 的 tensor parallel 切分方式对齐。

### 15.1 ColumnParallelLinearWithLoRA

Column parallel 的 base 权重按输出维切分。

LoRA A 不切：

```python
def slice_lora_a(self, lora_a):
    return lora_a
```

LoRA B 按输出维切：

```python
lora_b = lora_b[start_idx:end_idx, :]
```

位置：`lora/layers/column_parallel_linear.py:104` 到 `column_parallel_linear.py:130`

forward 时：

```text
local output = quantized column-parallel base + local LoRA B slice delta；
如果 base_layer.gather_output=True，再 all_gather。
```

位置：`lora/layers/column_parallel_linear.py:132` 到 `column_parallel_linear.py:158`

### 15.2 RowParallelLinearWithLoRA

Row parallel 的 base 权重按输入维切分。

LoRA A 按输入维切：

```python
lora_a = lora_a[:, start_idx:end_idx]
```

LoRA B 不切。

位置：`lora/layers/row_parallel_linear.py:32` 到 `row_parallel_linear.py:40`

forward 时：

```text
先构造 input_parallel；
再执行 quantized row-parallel base + LoRA delta；
最后按 base_layer.reduce_results 决定是否 all_reduce。
```

位置：`lora/layers/row_parallel_linear.py:42` 到 `row_parallel_linear.py:82`

### 15.3 fully_sharded_loras

`fully_sharded_loras=True` 时会启用 sharded LoRA wrapper。

例如：

```text
ColumnParallelLinearWithShardedLoRA
RowParallelLinearWithShardedLoRA
MergedColumnParallelLinearWithShardedLoRA
```

这些 wrapper 会额外切 LoRA rank 或输出维，减少单卡 LoRA 计算 / 存储压力。

但它会和 dual stream、部分 MoE EP 组合产生限制，后面会展开。

---

## 16. 量化 Linear 与 LoRA 的兼容边界

普通 Linear 的兼容关键是：

```text
LoRA wrapper 调用的是 base_layer.quant_method.apply(base_layer, x, bias)。
```

位置：`lora/layers/base_linear.py:204` 到 `base_linear.py:208`

因此只要量化方法满足：

```text
1. base_layer 是 LinearBase 派生类；
2. base_layer.quant_method 存在；
3. quant_method.apply() 返回正常 output tensor；
4. output dtype / shape 可以被 LoRA delta 加上；
```

LoRA 就可以在输出侧叠加。

这也是为什么 `BaseLinearLayerWithLoRA.weight` 要兼容多种量化参数名：

```text
weight：普通未量化；
weight_packed：compressed-tensors；
qweight：GPTQ / AWQ；
B：marlin。
```

位置：`lora/layers/base_linear.py:306` 到 `base_linear.py:321`

这个属性主要给外部访问权重时使用；forward 本身仍然走 `quant_method.apply()`。

---

## 17. async dual stream：base 量化 GEMM 和 LoRA delta 可以重叠

如果启用 `VLLM_LORA_ENABLE_DUAL_STREAM`，普通 Linear wrapper 可能走：

```python
_apply_async_impl()
```

位置：`lora/layers/base_linear.py:240` 到 `base_linear.py:304`

它拆成两个函数：

```python
base_fn():
    return self._get_quant_method().apply(self.base_layer, x, bias)

lora_fn():
    self.punica_wrapper.add_lora_linear(..., add_inputs=False)
```

位置：`lora/layers/base_linear.py:253` 到 `base_linear.py:278`

然后：

```python
output, lora_result = maybe_execute_in_parallel(...)
output.add_(lora_result)
```

位置：`lora/layers/base_linear.py:280` 到 `base_linear.py:297`

这说明：

```text
base 量化 GEMM 和 LoRA delta 不是必须串行；
在 CUDA-like 平台和配置允许时，可以用 aux stream 重叠执行。
```

限制是：

```text
dual stream 只支持 CUDA-like 平台；
fully_sharded_loras 会关闭 dual stream。
```

位置：`config/lora.py:117` 到 `config/lora.py:124`

---

## 18. MoE + LoRA + quantization 的路径更特殊

MoE 不只是简单在一个 Linear 输出上加 delta。

`FusedMoEWithLoRA` 包住的是：

```python
base_layer: MoERunner
```

位置：`lora/layers/fused_moe.py:33` 到 `fused_moe.py:37`

它会检查：

```python
assert not routed_experts.quant_method.is_monolithic
```

位置：`lora/layers/fused_moe.py:41` 到 `fused_moe.py:44`

这说明：

```text
monolithic MoE kernel 不支持 Fused MoE LoRA。
```

随后它拿到或构造 modular MoE kernel，并要求：

```python
assert moe_kernel.supports_lora()
```

位置：`lora/layers/fused_moe.py:64` 到 `fused_moe.py:88`

报错信息里也说明：

```text
For quantized MoE, mix LoRAExpertsMixin into the experts class and consume self._lora_context in apply().
```

也就是说：

```text
量化 MoE 要支持 LoRA，不能只实现普通 quantized experts；
它的 experts 实现还必须支持 LoRA context。
```

---

## 19. MoE LoRA 如何把 context 传入 quantized experts

`FusedMoEWithLoRA._build_lora_context()` 会构造：

```python
MoELoRAContext(...)
```

位置：`lora/layers/fused_moe.py:131` 到 `fused_moe.py:154`

里面包含：

```text
w13_lora_a_stacked / w13_lora_b_stacked
w2_lora_a_stacked / w2_lora_b_stacked
adapter_enabled
max_loras
top_k
fully_sharded
tp_rank / tp_size
local_num_experts
punica_wrapper
aux_stream / events
local_token_lora_mapping
original_hidden_states
```

`set_mapping()` 时会把 context 塞给 fused experts：

```python
fused_experts.set_lora_context(lora_context)
```

位置：`lora/layers/fused_moe.py:418` 到 `fused_moe.py:428`

`LoRAExpertsMixin` 提供：

```text
supports_lora() = True
apply_w13_lora(...)
apply_w2_lora(...)
```

位置：`model_executor/layers/fused_moe/experts/lora_experts_mixin.py:9` 到 `lora_experts_mixin.py:116`

它本质上把 MoE 的 LoRA delta 分成两段：

```text
w13 LoRA：加到第一段 expert GEMM 输出、activation 之前；
w2 LoRA：加到第二段 expert GEMM 输出、moe_sum 之前。
```

---

## 20. 量化 MoE 为什么要保存 original_hidden_states

`MoELoRAContext` 中有一个字段：

```python
original_hidden_states: torch.Tensor | None = None
```

注释写明：

```text
Original unquantized hidden states, stashed by the modular kernel
before the prepare step potentially quantizes them.
Used by apply_w13_lora so the LoRA kernel sees correct-magnitude activations
instead of raw quantized values that are missing the activation scale.
```

位置：`model_executor/layers/fused_moe/experts/lora_context.py:63` 到 `lora_context.py:68`

这就是 MoE + activation quantization + LoRA 的关键点：

```text
base MoE GEMM 可能需要量化后的 hidden_states；
LoRA A 侧计算应该使用原始未量化 hidden_states，避免直接吃缺少 scale 的 raw quantized value。
```

`triton_moe.py` 中 w13 LoRA 输入选择逻辑是：

```text
如果有 lora_unquantized_hidden_states，用它；
否则如果 context.original_hidden_states 行数匹配，用它；
否则退回 hidden_states。
```

位置：`model_executor/layers/fused_moe/experts/triton_moe.py:313` 到 `triton_moe.py:333`

这说明 vLLM 对 MoE LoRA + 量化不是简单相加，而是专门处理了：

```text
activation quantization 后 LoRA 输入尺度不一致的问题。
```

---

## 21. MoE LoRA 在 Triton experts 中如何叠加

`triton_moe.py` 中第一段 expert GEMM 后：

```text
base w13 GEMM → intermediate_cache1
LoRA w13 delta → intermediate_cache1
activation → intermediate_cache2
base w2 GEMM → intermediate_cache3
LoRA w2 delta → intermediate_cache3
moe_sum → output
```

源码位置：

```text
w13 LoRA：triton_moe.py:307 到 triton_moe.py:415
w2 LoRA：triton_moe.py:448 到 triton_moe.py:522
moe_sum：triton_moe.py:524 到 triton_moe.py:528
```

一个关键优化：

```text
当 lora_context is None 时，某些 FP8 block quant + SiLU fusion 可以启用；
当 LoRA 存在时，为了保留 BF16 intermediate 给 LoRA，相关融合会被跳过。
```

对应条件：

```python
and lora_context is None
```

位置：`model_executor/layers/fused_moe/experts/triton_moe.py:419` 到 `triton_moe.py:427`

这说明：

```text
LoRA + 量化可能影响 MoE 内部 fusion 和性能路径。
```

---

## 22. Fused MoE LoRA 的并行限制

### 22.1 monolithic kernel 不支持

`FusedMoEWithLoRA` 明确断言：

```python
assert not routed_experts.quant_method.is_monolithic
```

位置：`lora/layers/fused_moe.py:41` 到 `fused_moe.py:44`

所以如果某种量化 MoE 后端只能走 monolithic kernel，就不能直接挂 Fused MoE LoRA。

### 22.2 quantized experts 必须 supports_lora

`moe_kernel.supports_lora()` 不通过会报错。

位置：`lora/layers/fused_moe.py:82` 到 `fused_moe.py:88`

对量化 MoE 来说，专家实现要混入：

```text
LoRAExpertsMixin
```

并在 `apply()` 中消费 `_lora_context`。

### 22.3 EP + fully_sharded_loras 不兼容

`FusedMoEWithLoRA._verify_ep_fs()` 中：

```python
assert not (self.use_ep and lora_config.fully_sharded_loras)
```

位置：`lora/layers/fused_moe.py:228` 到 `fused_moe.py:236`

原因是：

```text
EP 和 fully sharded LoRA 都沿 TP group 做切分，
但一个按 expert 维，一个按 LoRA rank / hidden 维，假设冲突。
```

### 22.4 EP all2all backend 有限制

如果启用 expert parallel：

```python
assert all2all_backend == "allgather_reducescatter"
```

位置：`lora/layers/fused_moe.py:218` 到 `fused_moe.py:226`

---

## 23. LoRA adapter 加载与 base 量化权重没有共享存储

`LoRAModel.from_local_checkpoint()` 会读取：

```text
adapter_model.safetensors
adapter_model.bin
adapter_model.pt
```

位置：`lora/lora_model.py:205` 到 `lora_model.py:207`

它会跳过 base embedding weight：

```python
if is_base_embedding_weights(tensor_name):
    continue
```

位置：`lora/lora_model.py:132` 到 `lora_model.py:133`

LoRA 权重对象是：

```python
LoRALayerWeights(module_name, rank, lora_alpha, lora_a, lora_b, scaling)
```

位置：`lora/lora_weights.py:13` 到 `lora_weights.py:35`

这和 base quantized checkpoint 是两套权重流：

```text
base model loader 负责 qweight / scales / packed weight；
LoRA loader 负责 adapter A/B；
二者不在加载阶段合并。
```

---

## 24. 多 LoRA / hot-swap 和量化 base 如何共存

LoRA manager 有两层缓存：

```text
_registered_adapters：CPU 侧已注册 adapter；
_active_adapters / lora_index_to_id：GPU slot 中激活的 adapter。
```

位置：`lora/model_manager.py:101` 到 `model_manager.py:110`

`LRUCacheWorkerLoRAManager` 会在请求到来时：

```text
- 如果 adapter 不在 cache，先加载；
- 如果超过 max_cpu_loras，淘汰最老 adapter；
- 激活 adapter 到 GPU LoRA slot；
- 更新 Punica metadata。
```

位置：`lora/worker_manager.py:266` 到 `worker_manager.py:315`

量化 base model 不参与 hot-swap：

```text
base quantized weights 常驻；
LoRA A/B buffers 可替换；
token_lora_mapping 决定每个 token 用哪个 slot。
```

所以多 LoRA 的动态性来自 LoRA slots，而不是改写低 bit base 权重。

---

## 25. CUDA graph / compile 与 LoRA + 量化

LoRA 会影响 graph capture 的原因主要有两个。

### 25.1 LoRAConfig 进入 VllmConfig hash

`VllmConfig.compute_hash()` 会加入：

```python
self.lora_config.compute_hash()
```

位置：`config/vllm.py:443` 到 `config/vllm.py:446`

`LoRAConfig.compute_hash()` 包含：

```text
max_lora_rank
max_loras
fully_sharded_loras
lora_dtype
enable_tower_connector_lora
enable_mixed_moe_lora_format
target_modules
```

位置：`config/lora.py:81` 到 `config/lora.py:106`

这说明这些配置会影响模型执行图结构。

### 25.2 active LoRA 数量可专门 capture

`LoRAKernelMeta` 支持：

```text
captured_lora_counts
num_active_loras_cpu
```

位置：`lora/ops/triton_ops/lora_kernel_metadata.py:13` 到 `lora_kernel_metadata.py:47`

`get_captured_lora_counts()` 规则是：

```text
specialize=False：只 capture max_loras + 1；
specialize=True：capture 2 的幂以及 max_loras + 1。
```

位置：`lora/utils.py:49` 到 `lora/utils.py:64`

`+1` 是因为：

```text
no-LoRA token 也算一种可能的 active id。
```

这会影响启动时间、显存和不同 active LoRA 数量下的性能。

---

## 26. LoRA + 量化对显存的影响怎么理解

显存主要由两部分组成：

```text
quantized base model：
  qweight / scales / packed layout / workspace。

LoRA runtime buffers：
  max_loras × 每个可替换 layer 的 A/B stacked buffer。
```

普通 Linear 的 buffer 形状大致是：

```text
lora_a_stacked:
  (max_loras, 1, local_rank, input_size_or_partition)

lora_b_stacked:
  (max_loras, 1, output_size_or_partition, max_lora_rank)
```

位置：`lora/layers/base_linear.py:129` 到 `base_linear.py:150`

MoE 的 buffer 更大，因为带专家维：

```text
w13_lora_a_stacked:
  (max_loras, local_num_experts, rank, hidden_size)

w13_lora_b_stacked:
  (max_loras, local_num_experts, intermediate_size_per_partition, rank)

w2_lora_a_stacked:
  (max_loras, local_num_experts, rank, intermediate_size_per_partition)

w2_lora_b_stacked:
  (max_loras, local_num_experts, hidden_size_or_partition, rank)
```

位置：`lora/layers/fused_moe.py:156` 到 `fused_moe.py:216`

因此：

```text
量化 base model 会降低 base 权重显存；
但 LoRA 额外显存与 max_loras、max_lora_rank、target_modules、MoE expert 数、dtype 直接相关。
```

如果量化后显存仍然紧张，优先看：

```text
max_loras
max_lora_rank
lora_target_modules
fully_sharded_loras
MoE LoRA 是否覆盖所有 experts
```

---

## 27. 和量化方法相关的兼容性总结

### 27.1 普通 Linear 量化

只要量化方法实现了：

```text
QuantizeMethodBase.apply(layer, x, bias)
```

并且 layer 仍是 `LinearBase` 派生类，LoRA wrapper 通常可以叠加。

典型可共存路径：

```text
AWQ / GPTQ / Marlin / FP8 / compressed-tensors / online quant
  → LinearBase.quant_method.apply()
  → LoRA delta add
```

### 27.2 量化 MoE

要求更高：

```text
- 不能是 monolithic MoE kernel；
- modular MoE kernel 必须 supports_lora；
- quantized experts 实现要混入 LoRAExpertsMixin；
- activation quantization 下要正确处理 original_hidden_states；
- EP / all2all backend / fully sharded LoRA 有额外限制。
```

### 27.3 FP8 / activation quantization

普通 Linear LoRA 通常用原始 activation `x` 做 LoRA A/B，base quantized GEMM 内部自己处理 scale。

MoE 更复杂，源码专门保存未量化 hidden states，避免 LoRA 使用缺 scale 的 quantized activation。

位置：`model_executor/layers/fused_moe/experts/lora_context.py:63` 到 `lora_context.py:68`

---

## 28. 常见运行路径拆解

### 28.1 AWQ base + LoRA

```text
--quantization awq --enable-lora
  → base LinearBase.quant_method = AutoAWQLinearMethod / Marlin variant
  → LoRA manager 替换 LinearBase 为 Column/Row/Replicated WithLoRA
  → forward 先跑 AWQ quant_method.apply()
  → Punica 计算 LoRA delta
  → output.add_(delta)
```

重点：

```text
LoRA adapter 不会写入 AWQ qweight；
LoRA delta 在输出侧叠加。
```

### 28.2 GPTQ / Marlin base + LoRA

```text
--quantization gptq / gptq_marlin --enable-lora
  → base layer 可能持有 qweight 或 Marlin B
  → wrapper.weight 属性兼容 qweight / B
  → forward 仍通过 quant_method.apply()
  → LoRA delta 独立计算
```

关键源码：`lora/layers/base_linear.py:306` 到 `base_linear.py:321`

### 28.3 FP8 base + LoRA

```text
--quantization fp8 --enable-lora
  → base Linear / MoE 可能使用 FP8 quant method
  → LoRA A/B 默认仍是 fp16 / bf16
  → Linear 路径输出侧相加
  → MoE 路径需要 original_hidden_states / lora_context 处理 activation quant
```

### 28.4 quantized MoE + LoRA

```text
MoERunner
  → FusedMoEWithLoRA
  → build MoELoRAContext
  → quantized modular MoE experts apply()
  → w13 LoRA delta before activation
  → w2 LoRA delta before moe_sum
```

如果遇到不支持，通常报错会指向：

```text
Monolithic kernels are not supported for Fused MoE LoRA.
...
For quantized MoE, mix LoRAExpertsMixin into the experts class and consume self._lora_context in apply().
```

位置：`lora/layers/fused_moe.py:41` 到 `fused_moe.py:88`

---

## 29. 容易混淆的点

### 29.1 base model 量化是否意味着 LoRA adapter 也量化？

不是。

base 权重由 `quant_config / quant_method` 管；LoRA A/B 由 `LoRAConfig.lora_dtype` 管。

默认 LoRA dtype 是 base model activation dtype，而不是 base weight 的低 bit dtype。

### 29.2 LoRA 是否 merge 到 quantized weight 里？

不是运行时路径。

vLLM 的动态 LoRA 是：

```text
base output + LoRA delta
```

不是：

```text
quantized(W_base + ΔW_lora)
```

### 29.3 LoRA 会不会破坏量化 kernel？

普通 Linear 不会改 base quant_method，只是在外面包一层 wrapper。

但 MoE 量化 kernel 必须显式支持 LoRA context，否则不能共存。

### 29.4 为什么 packed layer 要特殊处理？

因为 runtime 里可能只有 `qkv_proj`，adapter 里却是 `q_proj/k_proj/v_proj`。

LoRA manager 必须把多个 adapter 权重重新 pack 成 runtime layer 对应的 `PackedLoRALayerWeights`。

### 29.5 LoRA rank 会不会影响 CUDA graph？

会。

`max_lora_rank`、`max_loras`、`fully_sharded_loras`、`target_modules` 都进入 `LoRAConfig.compute_hash()`，会影响图结构和 capture。

### 29.6 同一个 batch 能不能混合多个 LoRA 和 no-LoRA？

可以。

`token_lora_mapping` 支持每个 token 对应一个 LoRA slot，`-1` 表示 no-LoRA。

位置：`lora/punica_wrapper/punica_base.py:249` 到 `punica_base.py:256`

---

## 30. 调试时应该看哪些位置

如果想确认 LoRA 是否启用：

```text
engine/arg_utils.py:2145
config/vllm.py:895
v1/worker/lora_model_runner_mixin.py:31
```

如果想确认 LoRA dtype / rank / slots：

```text
config/lora.py:34
config/lora.py:108
config/lora.py:127
```

如果想确认哪些 layer 被替换成 LoRA wrapper：

```text
lora/model_manager.py:382
lora/utils.py:106
lora/utils.py:219
lora/utils.py:271
```

如果想确认量化 base forward 与 LoRA delta 如何相加：

```text
lora/layers/base_linear.py:186
lora/layers/base_linear.py:204
lora/layers/base_linear.py:215
lora/punica_wrapper/punica_gpu.py:203
```

如果想确认 packed QKV / gate_up LoRA 如何处理：

```text
lora/utils.py:371
lora/model_manager.py:718
lora/model_manager.py:731
lora/lora_weights.py:99
```

如果想确认 MoE + LoRA + quantization：

```text
lora/layers/fused_moe.py:33
lora/layers/fused_moe.py:82
model_executor/layers/fused_moe/experts/lora_context.py:11
model_executor/layers/fused_moe/experts/lora_experts_mixin.py:9
model_executor/layers/fused_moe/experts/triton_moe.py:307
```

如果想确认运行时 token→LoRA 映射：

```text
v1/worker/lora_model_runner_mixin.py:73
lora/model_manager.py:351
lora/punica_wrapper/punica_base.py:168
lora/ops/triton_ops/lora_kernel_metadata.py:109
```

---

## 31. 最小心智模型

如果只记一条：

```text
LoRA 和量化在 vLLM 里是“输出侧叠加”的关系：
量化负责 base layer 的权重存储和 base GEMM，
LoRA 负责按请求动态选择 adapter 并计算低秩 delta，
两者在 forward 输出中相加。
```

再压缩成一句话：

```text
QuantizationConfig 决定 base 怎么算，LoRAConfig 决定 adapter 怎么挂，Punica 决定当前 batch 的 LoRA delta 怎么批量加到量化 base output 上。
```
