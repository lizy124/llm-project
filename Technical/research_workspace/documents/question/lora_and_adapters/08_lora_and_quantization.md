# 08. LoRA 如何与量化 base model 共存？

源码位置：

- `code/vllm/vllm/config/lora.py`
- `code/vllm/vllm/lora/layers/base_linear.py`
- `code/vllm/vllm/lora/layers/column_parallel_linear.py`
- `code/vllm/vllm/lora/layers/row_parallel_linear.py`
- `code/vllm/vllm/lora/layers/fused_moe.py`
- `code/vllm/vllm/lora/layers/utils.py`
- `code/vllm/vllm/lora/lora_model.py`
- `code/vllm/vllm/lora/lora_weights.py`
- `code/vllm/vllm/lora/model_manager.py`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：量化 base model 与 LoRA adapter 同时启用时，权重存储、layer 包装、forward 顺序、dtype、kernel、CUDA graph 和兼容边界如何处理。

---

## 1. 一句话回答

LoRA + 量化在 vLLM 中通常表示：base model 的 Linear/MoE 权重按量化 backend 存储和执行；LoRA adapter 作为额外的低秩 A/B 权重，以 `LoRAConfig.lora_dtype` 保存，forward 时先计算量化 base output，再把 LoRA delta 加到同一个 output 上。

最小公式：

```text
final_output = quantized_base_layer(x) + lora_delta(x, active_lora_mapping)
```

主链路是：

```text
quantized base model load
  → LinearBase.quant_method = AWQ / GPTQ / FP8 / compressed tensors / ...
  → LoRAModelManager 把支持 LoRA 的 LinearBase 替换成 LoRA wrapper
  → LoRA wrapper.forward()
      → base_layer.quant_method.apply(base_layer, x, bias)
      → punica_wrapper.add_lora_linear / add_shrink / add_expand
      → output += LoRA delta
  → 后续 layer / logits / sampling
```

所以：

```text
量化负责 base layer 怎么算；
LoRA 负责 adapter delta 怎么叠加；
两者在 LoRA-wrapped layer 的 forward 中汇合。
```

---

## 2. base model 量化发生在哪里

vLLM 的 Linear 层通过 `LinearBase.quant_method` 抽象具体执行方式。

`LinearBase.__init__()` 中：

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
else:
    raise ValueError("All linear layers should support quant method.")
```

位置：`linear.py:242` 到 `linear.py:275`

`LinearMethodBase.apply()` 是量化 / 非量化 Linear 的统一执行入口：

```python
def apply(self, layer, x, bias=None) -> torch.Tensor:
    raise NotImplementedError
```

位置：`linear.py:138` 到 `linear.py:176`

非量化只是其中一种方法：

```python
class UnquantizedLinearMethod(LinearMethodBase):
    def apply(...):
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
```

位置：`linear.py:179` 到 `linear.py:225`

量化方法则在：

```text
code/vllm/vllm/model_executor/layers/quantization/
```

例如 AWQ、GPTQ、FP8、Marlin、compressed tensors 等，都会通过对应 `quant_method.apply()` 被调用。

---

## 3. LoRA layer 如何包住量化 Linear

LoRA 不替换 `quant_method`。它把原来的 `LinearBase` 包在 LoRA wrapper 的 `base_layer` 里。

`BaseLinearLayerWithLoRA.__init__()`：

```python
self.base_layer = base_layer
self.input_size = self.base_layer.input_size
self.tp_size = self.base_layer.tp_size
self.tp_rank = self.base_layer.tp_rank
self.device = _get_lora_device(self.base_layer)
```

位置：`base_linear.py:69` 到 `base_linear.py:80`

这说明：

```text
base_layer 仍然是原来的 LinearBase；
如果它是量化 Linear，量化权重仍在 base_layer 上；
LoRA wrapper 只是在外面加一层 adapter delta 逻辑。
```

`LoRAModelManager._create_lora_modules()` 负责替换 module：

```python
new_module = replace_submodule(
    self.model,
    module_name,
    from_layer(...),
)
```

位置：`model_manager.py:375` 到 `model_manager.py:502`

`from_layer()` 会选择合适的 LoRA wrapper：

位置：`utils.py:106` 到 `utils.py:124`

---

## 4. forward 顺序：先量化 base，再 LoRA delta

### 4.1 普通 Linear LoRA

`BaseLinearLayerWithLoRA._apply_sync()`：

```python
output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
return self._apply_lora_to_output(x, output)
```

位置：`base_linear.py:195` 到 `base_linear.py:199`

`_apply_lora_to_output()`：

```python
lora_output = self.punica_wrapper.add_lora_linear(
    output, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, self.output_slices
)
if not current_platform.can_update_inplace():
    output = lora_output
return output
```

位置：`base_linear.py:206` 到 `base_linear.py:229`

所以执行顺序明确是：

```text
x
  → base_layer.quant_method.apply()
  → base output
  → punica_wrapper.add_lora_linear(output, x, A, B, mapping)
  → final output
```

### 4.2 ColumnParallelLinear LoRA

Column parallel 的通用 `_mcp_apply()` 也是先调用量化 base：

```python
output = layer.base_layer.quant_method.apply(layer.base_layer, x, bias)
```

位置：`column_parallel_linear.py:24` 到 `column_parallel_linear.py:40`

然后执行 LoRA shrink / all-gather / expand：

```python
shrunk_buffers = layer.punica_wrapper.add_shrink(...)
buffers = tensor_model_parallel_all_gather(buffers)
lora_output = layer.punica_wrapper.add_expand(output, buffers, ...)
```

位置：`column_parallel_linear.py:41` 到 `column_parallel_linear.py:80`

### 4.3 RowParallelLinear fully-sharded LoRA

Fully sharded row parallel 路径同样先算 base：

```python
output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
```

位置：`row_parallel_linear.py:118` 到 `row_parallel_linear.py:120`

然后：

```text
add_shrink → all_reduce buffer → add_expand → output
```

位置：`row_parallel_linear.py:121` 到 `row_parallel_linear.py:159`

---

## 5. 量化权重和 LoRA 权重分别放在哪里

### 5.1 量化 base 权重在 base_layer 上

量化方式不同，base layer 上的权重属性也不同。

LoRA 工具函数 `_get_lora_device()` 为了找 LoRA tensor 应放在哪个设备，会识别多种 base layer 权重属性：

```python
if hasattr(base_layer, "weight"):
    return base_layer.weight.device
elif hasattr(base_layer, "weight_packed"):
    return base_layer.weight_packed.device
elif hasattr(base_layer, "qweight"):
    return base_layer.qweight.device
elif hasattr(base_layer, "w2_weight"):
    return base_layer.w2_weight.device
elif hasattr(base_layer, "w2_weight_packed"):
    return base_layer.w2_weight_packed.device
elif hasattr(base_layer, "w2_qweight"):
    return base_layer.w2_qweight.device
else:
    raise ValueError(...)
```

位置：`utils.py:45` 到 `utils.py:70`

对应关系：

```text
weight：
  非量化 Linear。

weight_packed：
  compressed tensors 等 packed 权重。

qweight：
  GPTQ / AWQ 类量化权重。

w2_weight / w2_weight_packed / w2_qweight：
  MoE 层对应的非量化 / packed / quantized 权重。
```

这说明 LoRA wrapper 不直接解码量化权重；它只需要知道 base layer 的设备，真正量化计算由 `quant_method.apply()` 完成。

### 5.2 LoRA 权重在 lora_a_stacked / lora_b_stacked 上

LoRA wrapper 初始化时预分配 LoRA slot。

普通 Linear：

```python
self.lora_a_stacked = tuple(torch.zeros(max_loras, 1, ..., dtype=lora_config.lora_dtype, device=self.device))
self.lora_b_stacked = tuple(torch.zeros(max_loras, 1, ..., dtype=lora_config.lora_dtype, device=self.device))
```

位置：`base_linear.py:99` 到 `base_linear.py:150`

Column parallel：

位置：`column_parallel_linear.py:205` 到 `column_parallel_linear.py:244`

这些 stacked tensor 的含义是：

```text
第 0 维：LoRA slot index；
后续维度：rank/input/output 维度；
dtype：LoRAConfig.lora_dtype；
device：跟随 base layer 设备。
```

adapter 激活时，权重会被 copy 进去：

```python
self.lora_a_stacked[...][index, ...].copy_(lora_a, non_blocking=True)
self.lora_b_stacked[...][index, ...].copy_(lora_b, non_blocking=True)
```

位置：`base_linear.py:157` 到 `base_linear.py:183`，`column_parallel_linear.py:300` 到 `column_parallel_linear.py:326`

---

## 6. LoRA dtype 如何确定

`LoRAConfig.lora_dtype`：

```python
lora_dtype: torch.dtype | LoRADType = "auto"
```

位置：`config/lora.py:46` 到 `config/lora.py:47`

`verify_with_model_config()`：

```python
if self.lora_dtype in (None, "auto"):
    self.lora_dtype = model_config.dtype
elif isinstance(self.lora_dtype, str):
    self.lora_dtype = getattr(torch, self.lora_dtype)
```

位置：`config/lora.py:127` 到 `config/lora.py:131`

加载 checkpoint tensor 时：

```python
loras[module_name].lora_a = tensor.to(device=device, dtype=dtype)
loras[module_name].lora_b = tensor.to(device=device, dtype=dtype)
```

位置：`lora_model.py:155` 到 `lora_model.py:162`

预分配 slot 时：

```python
dtype=lora_config.lora_dtype
```

位置：`base_linear.py:128` 到 `base_linear.py:149`

所以通常情况是：

```text
base model 可以是 int4/int8/fp8/packed 量化；
LoRA A/B 通常是 fp16 或 bf16；
"auto" 时跟随 model_config.dtype，而不是跟随 base quantized storage dtype。
```

这就是“量化 base + 非量化 LoRA delta”的常见组合。

---

## 7. LoRA adapter 本身是否量化？

从当前主路径看，vLLM 的 LoRA 权重加载是：

```text
checkpoint tensor → tensor.to(dtype=LoRAConfig.lora_dtype) → lora_a/lora_b
```

位置：`lora_model.py:117` 到 `lora_model.py:164`

`LoRALayerWeights` 保存的是普通 tensor：

```python
lora_a: torch.Tensor
lora_b: torch.Tensor
```

位置：`lora_weights.py:13` 到 `lora_weights.py:35`

执行时 LoRA delta 通过 punica wrapper 的 LoRA kernel 计算。

因此要区分：

```text
base model quantization：
  vLLM 支持多种低 bit / packed / fp8 base 权重。

LoRA adapter quantization：
  当前主路径不是把 LoRA A/B 当作 AWQ/GPTQ base weight 那样量化存储，
  而是按 lora_dtype 作为额外低秩权重执行。
```

即：

```text
量化 base model 不等于量化 LoRA adapter。
```

---

## 8. 哪些量化 Linear 可以和 LoRA 共存

LoRA wrapper 能否包住一个 layer，主要看两件事：

```text
1. 这个 layer 类型是否是 vLLM LoRA wrapper 支持的类型；
2. 这个 base layer 是否有可用 quant_method.apply() 和可识别的权重设备属性。
```

`_all_lora_classes` 包含：

```text
VocabParallelEmbeddingWithLoRA
ColumnParallelLinearWithLoRA
MergedColumnParallelLinearWithLoRA
QKVParallelLinearWithLoRA
MergedQKVParallelLinearWithLoRA
RowParallelLinearWithLoRA
ReplicatedLinearWithLoRA
LogitsProcessorWithLoRA
... fully sharded variants
FusedMoEWithLoRA
FusedMoE3DWithLoRA
```

位置：`utils.py:76` 到 `utils.py:95`

这些 wrapper 判断的是 `ColumnParallelLinear`、`RowParallelLinear`、`QKVParallelLinear`、`MoERunner` 等 vLLM layer 类型，而不是某个量化算法名。

因为量化被封装在：

```text
base_layer.quant_method.apply()
```

只要量化 backend 实现了这个接口，并且 layer 类型能被 LoRA wrapper 替换，就可以走 LoRA + 量化组合。

---

## 9. WEIGHT_LOADER_V2_SUPPORTED 和 LoRA 的关系

`linear.py` 中列出支持 weight_loader_v2 的 LinearMethod：

```python
WEIGHT_LOADER_V2_SUPPORTED = [
    "UnquantizedLinearMethod",
    "CompressedTensorsLinearMethod",
    "CompressedTensorsLinearTransformMethod",
    "AWQMarlinLinearMethod",
    "AWQLinearMethod",
    "AutoGPTQLinearMethod",
    "Fp8LinearMethod",
    "FBGEMMFp8LinearMethod",
    "ModelOptFp8LinearMethod",
    "ModelOptFp8PcPtLinearMethod",
    "ModelOptFp8PbWoLinearMethod",
    "QuarkLinearMethod",
    "ModelOptNvFp4LinearMethod",
    "ModelOptNvFp4W4A16LinearMethod",
    "HummingLinearMethod",
]
```

位置：`linear.py:45` 到 `linear.py:61`

这说明 vLLM 的 base model 权重加载层面知道很多量化 LinearMethod。

但 LoRA 是否能叠加，不是仅看这个列表，而是看：

```text
- 最终 layer 是不是 LoRA wrapper 支持的 Linear/MoE/Embedding/LM head 类型；
- quant_method.apply() 能否返回可加 LoRA delta 的 output；
- LoRA wrapper 能否找到 base layer device；
- 对应平台是否支持 LoRA kernel。
```

---

## 10. base output 和 LoRA output 的 dtype / shape 对齐

LoRA wrapper 的策略是：

```text
1. base quant_method.apply() 产生 output；
2. LoRA kernel 以 output 和 x 为输入；
3. LoRA delta 加到 output 上；
4. 返回与 base output 相同语义的 tensor。
```

普通 Linear `_apply_lora_to_output()` 会处理 3D input：

```python
if x.ndim == 3 and output.ndim == 3:
    output = output.flatten(0, 1)
    x = x.flatten(0, 1)
...
if original_shape is not None:
    output = output.reshape(original_shape)
```

位置：`base_linear.py:206` 到 `base_linear.py:229`

原因是 punica 通常按二维 token 矩阵处理：

```text
(seq_len, hidden_dim)
```

但某些后端 / 多模态 encoder 可能给三维输入：

```text
(batch, seq_len, hidden_dim)
```

所以 wrapper 会 flatten 后加 LoRA，再 reshape 回原形状。

---

## 11. scaling 如何处理

LoRA 权重对象中保存 scaling：

```python
self.scaling = self.lora_alpha / self.rank
```

位置：`lora_weights.py:31` 到 `lora_weights.py:35`

如果是 rsLoRA，`PEFTHelper` 会使用：

```text
lora_alpha / sqrt(r)
```

位置：`peft_helper.py:53` 到 `peft_helper.py:59`

`LoRALayerWeights.optimize()` 会把 scaling 合并进 `lora_b`：

```python
self.lora_b *= self.scaling
self.scaling = 1
```

位置：`lora_weights.py:36` 到 `lora_weights.py:42`

`PackedLoRALayerWeights.optimize()` 对 packed 子 LoRA 也做类似处理：

位置：`lora_weights.py:230` 到 `lora_weights.py:237`

`LoRAModelManager._create_merged_loras_inplace()` 会对 lora 权重调用 optimize：

```python
for lora in lora_model.loras.values():
    lora.optimize()
```

位置：`model_manager.py:773` 到 `model_manager.py:775`

这意味着执行时 LoRA kernel 通常不再单独处理原始 alpha/r scaling，而是使用已经合并过 scaling 的 `lora_b`。

---

## 12. fully_sharded_loras 与量化 base 的关系

`LoRAConfig.fully_sharded_loras` 控制 LoRA A/B 在 TP 下如何切分：

```python
fully_sharded_loras: bool = False
```

位置：`config/lora.py:38` 到 `config/lora.py:42`

这个参数不是 base model 量化参数，而是 LoRA delta 的并行策略。

例如普通 Linear 创建 LoRA 权重时：

```python
lora_a_out_size = max_lora_rank
# 或 fully_sharded 时 divide(max_lora_rank, tp_size)
```

位置：`base_linear.py:99` 到 `base_linear.py:150`

RowParallel fully sharded 还会切 `lora_b`：

位置：`row_parallel_linear.py:101` 到 `row_parallel_linear.py:177`

所以：

```text
base quantization 决定 base_layer.quant_method；
fully_sharded_loras 决定 LoRA delta 在 TP 下如何分摊。
```

二者是两个不同维度。

---

## 13. CUDA graph / compile 与动态 LoRA

LoRA 会影响 CUDA graph 捕获，因为 active LoRA 数量和 mapping 会影响 kernel metadata。

`LoRAConfig.specialize_active_lora`：

```text
按 active LoRA adapter 数量构造 lora kernel grid；
开启后会为不同 active LoRA 数量捕获不同 CUDA graph；
可能提升变动 LoRA 使用模式的性能，但增加启动时间和内存。
```

位置：`config/lora.py:67` 到 `config/lora.py:73`

`get_captured_lora_counts()`：

```python
if not specialize:
    return [max_loras + 1]
return [n for n in range(1, max_loras + 2) if power_of_2_or_max]
```

位置：`utils.py:49` 到 `utils.py:64`

`GPUModelRunner` 在 batch descriptor 中也会计算 LoRA 状态：

```python
num_active_loras = len(self.input_batch.lora_id_to_lora_request)
has_lora = num_active_loras > 0
```

位置：`gpu_model_runner.py:3845` 到 `gpu_model_runner.py:3851`

profiling / dummy run 时会创建 dummy LoRA，保证图捕获和内存评估覆盖 LoRA 路径：

位置：`lora_model_runner_mixin.py:93` 到 `lora_model_runner_mixin.py:268`，`gpu_model_runner.py:5878` 到 `gpu_model_runner.py:5884`

---

## 14. 双 CUDA stream LoRA

LoRA 支持一个可选的双流执行模式：

```text
VLLM_LORA_ENABLE_DUAL_STREAM
```

`LoRAConfig` 校验：

```python
if envs.VLLM_LORA_ENABLE_DUAL_STREAM and not current_platform.is_cuda_alike():
    raise ValueError("Dual CUDA streams are only supported on CUDA platforms.")
if envs.VLLM_LORA_ENABLE_DUAL_STREAM and self.fully_sharded_loras:
    logger.warning_once(...)
    envs.VLLM_LORA_ENABLE_DUAL_STREAM = False
```

位置：`config/lora.py:117` 到 `config/lora.py:124`

`BaseLinearLayerWithLoRA.apply()` 中：

```python
if self._enable_aux_cuda_stream and is_forward_context_available():
    return torch.ops.vllm.lora_linear_async(...)
else:
    return self._apply_sync(x, bias)
```

位置：`base_linear.py:185` 到 `base_linear.py:193`

异步路径中，base layer 和 LoRA delta 可并行执行：

```text
base_fn：base_layer.quant_method.apply(...)
lora_fn：punica_wrapper.add_lora_linear(..., add_inputs=False)
output.add_(lora_result)
```

位置：`base_linear.py:231` 到 `base_linear.py:295`

这仍然不改变语义：

```text
final output = base output + LoRA delta
```

只是尝试让两部分计算重叠。

---

## 15. MoE 量化与 LoRA

LoRA 对 MoE 有专门 wrapper：

```text
FusedMoEWithLoRA
FusedMoE3DWithLoRA
```

位置：`utils.py:93` 到 `utils.py:95`

`_get_lora_device()` 识别 MoE 权重属性：

```text
w2_weight
w2_weight_packed
w2_qweight
```

位置：`utils.py:60` 到 `utils.py:68`

MoE LoRA kernel config 中有一条重要注释：

```text
LoRA shrink/expand operates on bf16/fp16 adapters regardless of the
base MoE weight's block-wise quantization.
```

位置：`utils.py:101` 到 `utils.py:129`

含义是：

```text
MoE base experts 可以是 block-wise quantized；
LoRA A/B delta 仍按 LoRA dtype 执行；
LoRA kernel 配置不把 base MoE 的 block_shape 当作 LoRA shrink/expand 的配置因素。
```

---

## 16. 性能和显存开销在哪里

启用 LoRA + 量化后，base model 显存会因量化降低，但 LoRA 仍会增加额外资源：

```text
1. 每个 LoRA-wrapped layer 预分配 lora_a_stacked / lora_b_stacked；
2. 大小与 max_loras、max_lora_rank、target modules、hidden sizes 有关；
3. 每轮 forward 额外执行 LoRA shrink / expand 或 add_lora_linear；
4. batch 内 active LoRA 数会影响 LoRA kernel metadata / graph capture；
5. CPU cache 还保存已注册 adapter 的 LoRAModel 权重。
```

关键配置：

```text
max_loras：
  GPU 同时活跃 LoRA slot 数，直接影响预分配 LoRA buffer。

max_lora_rank：
  每个 LoRA slot 的最大 rank，直接影响 A/B buffer 宽度。

max_cpu_loras：
  CPU 侧最多缓存多少 LoRAModel。

target_modules：
  限制哪些 module 注入 LoRA，可降低开销。

specialize_active_lora：
  可能提升运行性能，但增加图捕获数量和启动/内存成本。
```

位置：`config/lora.py:34` 到 `config/lora.py:79`

---

## 17. 常见误区

### 17.1 base model 量化是否表示 LoRA adapter 也量化？

不是。

base model 的权重由 quantization backend 管理；LoRA A/B 由 `LoRAConfig.lora_dtype` 控制，通常是 fp16/bf16。

### 17.2 某量化方式支持普通 forward，是否一定支持 LoRA？

不一定。

还要看：

```text
- 最终 layer 类型是否能被 LoRA wrapper 替换；
- wrapper 是否能找到 base layer device；
- quant_method.apply() 输出是否能被 LoRA delta 原地或非原地相加；
- 平台是否支持对应 LoRA kernel。
```

### 17.3 LoRA 会不会反量化 base weight 再相加？

从当前主路径看，不会。

LoRA wrapper 只调用：

```text
base_layer.quant_method.apply(...)
```

拿到 base output 后加 LoRA delta。它不直接读取、反量化、修改 base quantized weight。

### 17.4 LoRA rank 越大是否只影响 adapter 文件大小？

不是。

`max_lora_rank` 会影响：

```text
- LoRA slot 预分配大小；
- LoRA shrink/expand kernel 的计算量；
- CUDA graph / profiling 覆盖的内存；
- TP fully sharded 下 rank 切分合法性。
```

### 17.5 量化节省的显存是否会被 LoRA 完全抵消？

通常不会完全抵消，但取决于：

```text
max_loras × max_lora_rank × target_modules × hidden size
```

如果 `max_loras` 和 `max_lora_rank` 设置过大，LoRA buffer 和 CPU cache 会成为明显开销。

---

## 18. 调试时看哪些位置

### 18.1 确认 LoRA 是否启用

看：

```text
VllmConfig.lora_config 是否为 None
```

模型加载时：

```python
if self.lora_config:
    self.model = self.load_lora_model(...)
```

位置：`gpu_model_runner.py:5167` 到 `gpu_model_runner.py:5170`

### 18.2 确认 base layer 是否量化

看对应 LinearBase 的：

```text
base_layer.quant_method
base_layer.quant_config
```

位置：`linear.py:242` 到 `linear.py:275`

### 18.3 确认 LoRA 是否包住了 layer

看 LoRAModelManager：

```text
self.modules: dict[str, BaseLayerWithLoRA]
```

位置：`model_manager.py:106` 到 `model_manager.py:108`

### 18.4 确认 adapter 是否 active

看：

```text
lora_index_to_id
_active_adapters
```

位置：`model_manager.py:94` 到 `model_manager.py:104`

### 18.5 确认 token 使用哪个 LoRA

看 `InputBatch.make_lora_inputs()` 生成的：

```text
prompt_lora_mapping
token_lora_mapping
active_lora_requests
```

位置：`gpu_input_batch.py:976` 到 `gpu_input_batch.py:999`

---

## 19. 最小心智模型

LoRA 与量化共存可以分成三层：

```text
第一层：base quantization
  LinearBase.quant_method 决定 base layer 如何使用低 bit / packed / fp8 权重计算 output。

第二层：LoRA adapter
  LoRA A/B 以 lora_dtype 保存，按 adapter slot 预分配并在激活时写入。

第三层：runtime mapping
  LoRAMapping 告诉 punica wrapper 每个 token 用哪个 LoRA slot，
  LoRA wrapper 把 delta 加到 base output 上。
```

如果只记住一句话：

```text
量化和 LoRA 在 vLLM 里不是互相替代的机制：量化压缩并执行 base layer，LoRA 以额外低秩权重计算 delta，二者在 LoRA-wrapped layer 的 forward 中通过 base output + LoRA delta 汇合。
```
