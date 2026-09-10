# 02. 用户量化配置如何变成 QuantizationConfig？

源码位置：

- `code/vllm/vllm/engine/arg_utils.py`
- `code/vllm/vllm/config/model.py`
- `code/vllm/vllm/config/quantization.py`
- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/config/cache.py`
- `code/vllm/vllm/config/load.py`
- `code/vllm/vllm/model_executor/model_loader/weight_utils.py`
- `code/vllm/vllm/model_executor/model_loader/utils.py`
- `code/vllm/vllm/model_executor/layers/quantization/__init__.py`
- `code/vllm/vllm/model_executor/layers/quantization/base_config.py`
- `code/vllm/vllm/platforms/interface.py`

本问题只关注：用户传入的 `--quantization`、`--quantization-config`、checkpoint 自带 `quantization_config` / `compression_config`、`load_format`、`kv_cache_dtype` 如何在配置阶段被解析和校验，最终变成 `VllmConfig.quant_config`。至于 `quant_config` 如何作用到 Linear / MoE / Attention，在后续专题展开。

---

## 1. 一句话回答

用户量化配置会先进入 `EngineArgs`，再写入 `ModelConfig`，由 `ModelConfig._verify_quantization()` 确定量化方法名，最后由 `VllmConfig._get_quantization_config()` 调用 `get_quant_config()` 读取 checkpoint / 用户配置并构造具体 `QuantizationConfig` 对象。

最小链路是：

```text
CLI / API / checkpoint metadata
  → EngineArgs
  → resolve_quantization_config()
  → ModelConfig.quantization / ModelConfig.quantization_config
  → ModelConfig._verify_quantization()
  → VllmConfig._get_quantization_config()
  → get_quant_config()
  → get_quantization_config(method).from_config(...)
  → VllmConfig.quant_config
```

三个关键层次：

```text
ModelConfig.quantization：
  方法名字符串，例如 awq / gptq / fp8 / online。

ModelConfig.quantization_config：
  用户侧在线量化配置，类型可以是 QuantizationConfigArgs。

VllmConfig.quant_config：
  真正传给模型和 layer 的 QuantizationConfig 实例。
```

---

## 2. 配置阶段的对象关系

```text
EngineArgs
  ├─ quantization
  ├─ quantization_config
  ├─ load_format
  ├─ kv_cache_dtype
  ├─ calculate_kv_scales
  └─ kv_cache_dtype_skip_layers
       ↓
ModelConfig
  ├─ quantization
  ├─ quantization_config
  └─ allow_deprecated_quantization
       ↓
CacheConfig
  ├─ cache_dtype
  ├─ calculate_kv_scales
  └─ kv_cache_dtype_skip_layers
       ↓
LoadConfig
  └─ load_format
       ↓
VllmConfig
  └─ quant_config
```

这说明：

```text
权重量化配置、KV cache dtype 配置、权重加载格式配置，是三条相关但不同的配置线。
```

---

## 3. EngineArgs：用户配置从哪里进入

`EngineArgs` 中和量化相关的字段包括：

```text
quantization
quantization_config
load_format
kv_cache_dtype
calculate_kv_scales
kv_cache_dtype_skip_layers
```

位置：

- `code/vllm/vllm/engine/arg_utils.py:440`
- `code/vllm/vllm/engine/arg_utils.py:540`
- `code/vllm/vllm/engine/arg_utils.py:541`
- `code/vllm/vllm/engine/arg_utils.py:680`
- `code/vllm/vllm/engine/arg_utils.py:681`

### 3.1 先解析在线量化配置

`EngineArgs.__post_init__()` 会调用：

```python
self.quantization_config = resolve_quantization_config(
    self.quantization, self.quantization_config
)
```

位置：`code/vllm/vllm/engine/arg_utils.py:748`

这一步只处理：

```text
--quantization 的在线量化 shorthand
--quantization-config 的用户侧覆盖规则
```

它不会读取 checkpoint 文件。

### 3.2 再创建 ModelConfig / CacheConfig / LoadConfig

创建 `ModelConfig` 时会传入：

```text
quantization
quantization_config
allow_deprecated_quantization
```

位置：`code/vllm/vllm/engine/arg_utils.py:1617`

创建 `CacheConfig` 前，会先解析：

```python
resolve_kv_cache_dtype_string(self.kv_cache_dtype, model_config)
```

位置：`code/vllm/vllm/engine/arg_utils.py:1832`

这一步把用户侧 `--kv-cache-dtype auto` 解析成更明确的 cache dtype。

---

## 4. `resolve_quantization_config()` 做什么

入口在：`code/vllm/vllm/config/quantization.py:147`

它处理的是用户侧在线量化配置。

### 4.1 QuantSpec / QuantizationConfigArgs

核心结构：

```text
QuantSpec：
  weight
  activation

QuantizationConfigArgs：
  linear
  moe
  ignore
```

位置：`code/vllm/vllm/config/quantization.py:63`

`linear` 应用于 LinearBase，`moe` 应用于 FusedMoE / RoutedExperts，`ignore` 用来跳过指定 layer。

### 4.2 在线量化 shorthand

支持的 shorthand：

```text
fp8_per_tensor
fp8_per_block
fp8_per_channel
mxfp8
int8_per_channel_weight_only
online
```

位置：`code/vllm/vllm/config/quantization.py:112`

例如：

```text
--quantization fp8_per_tensor
```

会展开成：

```text
QuantizationConfigArgs(
  linear=QuantSpec(weight=kFp8StaticTensorSym),
  moe=QuantSpec(weight=kFp8StaticTensorSym),
)
```

### 4.3 合并规则

`resolve_quantization_config()` 的规则：

```text
1. 如果 quantization 是传统 checkpoint 方法：
   - 不允许再传用户侧 quantization_config；
   - 返回 None。

2. 如果 quantization 是在线量化 shorthand：
   - 先生成 base QuantizationConfigArgs。

3. 如果同时传了 quantization_config：
   - 用户显式字段覆盖 shorthand；
   - 未显式字段继承 shorthand。

4. 如果 quantization == "online"：
   - 依赖 quantization_config 提供完整规则。
```

这一步输出的是：

```text
ModelConfig.quantization_config
```

而不是最终的 `QuantizationConfig` 对象。

---

## 5. ModelConfig：确定量化方法名

`ModelConfig` 里保存：

```python
quantization: QuantizationMethods | str | None = None
quantization_config: dict[str, Any] | QuantizationConfigArgs | None = None
allow_deprecated_quantization: bool = False
```

位置：`code/vllm/vllm/config/model.py:197`

`quantization` 会先统一成小写。

位置：`code/vllm/vllm/config/model.py:749`

### 5.1 checkpoint 自带量化配置

`ModelConfig._verify_quantization()` 会读取：

```python
quant_cfg = self.model_arch_config.quantization_config
```

位置：`code/vllm/vllm/config/model.py:975`

如果 checkpoint 有：

```json
{
  "quantization_config": {
    "quant_method": "awq"
  }
}
```

且用户没有显式传 `--quantization`，则 vLLM 会把：

```text
self.quantization = quant_method
```

如果用户传入的 `--quantization` 和 checkpoint 的 `quant_method` 不一致，会报错。

位置：`code/vllm/vllm/config/model.py:1038`

### 5.2 override_quantization_method

有些 checkpoint 的 `quant_method` 只是存储格式，vLLM 运行时可能要改成更合适的方法。

`_verify_quantization()` 会遍历若干量化 config，并调用：

```python
method.override_quantization_method(
    quant_cfg, self.quantization, hf_config=self.hf_config
)
```

位置：`code/vllm/vllm/config/model.py:1014`

典型原因：

```text
GPTQ checkpoint 可能改用 gptq_marlin；
AWQ checkpoint 可能改用 awq_marlin；
ModelOpt / MXFP4 / DeepSeek V4 FP8 需要特殊识别；
用户指定 humming 时要优先走 humming。
```

### 5.3 方法名校验

`_verify_quantization()` 会检查：

```text
1. 是否在 QUANTIZATION_METHODS 中；
2. 当前 platform 是否支持；
3. 是否是 deprecated quantization。
```

位置：

- `code/vllm/vllm/config/model.py:1050`
- `code/vllm/vllm/config/model.py:1056`
- `code/vllm/vllm/config/model.py:1058`

这一步完成后，`ModelConfig.quantization` 才是 vLLM 认可的方法名。

---

## 6. QuantizationMethods 注册表

方法名列表在：`code/vllm/vllm/model_executor/layers/quantization/__init__.py:12`

常见方法包括：

```text
awq / auto_awq / awq_marlin
auto_gptq / gptq / gptq_marlin
fp8 / fbgemm_fp8 / fp_quant
compressed-tensors
bitsandbytes
modelopt / modelopt_fp4 / modelopt_mxfp8 / modelopt_mixed
quark / torchao / inc
mxfp4 / gpt_oss_mxfp4 / deepseek_v4_fp8
online / fp8_per_tensor / fp8_per_block / fp8_per_channel / mxfp8
```

字符串到配置类的映射在：

```python
get_quantization_config(quantization)
```

位置：`code/vllm/vllm/model_executor/layers/quantization/__init__.py:108`

例如：

```text
"awq"                → AutoAWQConfig
"gptq"               → AutoGPTQConfig
"fp8"                → Fp8Config
"compressed-tensors" → CompressedTensorsConfig
"bitsandbytes"       → BitsAndBytesConfig
"online"             → OnlineQuantizationConfig
```

---

## 7. VllmConfig：构造真正的 quant_config

`VllmConfig` 上有：

```python
quant_config: QuantizationConfig | None = None
```

位置：`code/vllm/vllm/config/vllm.py:334`

初始化时会调用：

```python
VllmConfig._get_quantization_config(model_config, load_config)
```

位置：`code/vllm/vllm/config/vllm.py:910`

### 7.1 _get_quantization_config 流程

入口：`code/vllm/vllm/config/vllm.py:609`

它做：

```text
1. 如果 model_config.quantization is None：
   返回 None。

2. 调用 get_quant_config(model_config, load_config)：
   构造 QuantizationConfig 实例。

3. 获取当前 device capability。

4. 校验 capability >= quant_config.get_min_capability()。

5. 校验 model_config.dtype in quant_config.get_supported_act_dtypes()。

6. 调用 quant_config.maybe_update_config(model_name, hf_config)。

7. 返回 quant_config。
```

位置：`code/vllm/vllm/config/vllm.py:615` 到 `code/vllm/vllm/config/vllm.py:641`

这一步之后，`VllmConfig.quant_config` 才真正可供模型初始化使用。

---

## 8. get_quant_config 如何读取配置

`get_quant_config()` 定义在：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:240`

它先用方法名找到配置类：

```python
quant_cls = get_quantization_config(model_config.quantization)
```

然后按优先级读取配置。

### 8.1 HF config / compression_config

优先读取：

```text
model_config.hf_config.quantization_config
model_config.hf_config.text_config.quantization_config
model_config.hf_config.compression_config
```

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:247`

如果存在且完整：

```python
return quant_cls.from_config(hf_quant_config)
```

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:275`

compressed-tensors 会额外补：

```text
total_num_heads
total_num_kv_heads
```

用于 TP-aware attention scale 加载。

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:257`

### 8.2 hf_overrides

如果 HF config 不足，会尝试：

```text
hf_overrides["quantization_config_file"]
hf_overrides["quantization_config_dict_json"]
```

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:293`

### 8.3 在线量化

如果 `model_config.quantization_config` 不为空，会构造：

```python
OnlineQuantizationConfig(args=model_config.quantization_config)
```

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:322`

注释说明：

```text
Online quantization doesn't read from checkpoint configs - it quantizes fp16/bf16 weights on the fly during loading.
```

### 8.4 bitsandbytes 特殊路径

如果方法是 `bitsandbytes`，可用空配置：

```python
return quant_cls.from_config({})
```

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:333`

### 8.5 量化配置文件

最后尝试在模型目录找量化方法声明的配置文件：

```python
quant_cls.get_config_filenames()
```

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:360`

例如 GPTQ 通常查找：

```text
quantize_config.json
```

---

## 9. QuantizationConfig 抽象负责什么

抽象类在：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:77`

它负责描述某个量化方法的配置和能力：

```text
get_name()
  返回方法名。

get_supported_act_dtypes()
  返回支持的 activation dtype。

get_min_capability()
  返回最低 GPU capability。

get_config_filenames()
  返回量化配置文件名。

from_config(config)
  从 checkpoint / json 配置构造对象。

override_quantization_method(...)
  根据 checkpoint 元数据修正最终方法名。

get_quant_method(layer, prefix)
  为具体 layer 选择 quant_method。

get_cache_scale_mapper()
  提供 KV cache scale 名称映射。

maybe_update_config(...)
  根据模型名 / HF config 做后处理更新。
```

本文只说明这些接口如何在配置阶段被构造；具体 `get_quant_method()` 如何影响 Linear / MoE / Attention，见 `04_quantized_linear_layers.md`、`07_kv_cache_quantization.md`、`09_moe_quantization.md`。

---

## 10. load_format 和 quantization 的边界

`LoadConfig.load_format` 在：`code/vllm/vllm/config/load.py:28`

它控制“用什么 loader / 文件格式读权重”，例如：

```text
auto
pt
safetensors
npcache
tensorizer
runai_streamer
bitsandbytes
sharded_state
mistral
```

它不是量化方法本身。

边界是：

```text
load_format：
  怎么读文件。

quantization：
  读出来的 tensor 如何解释、如何创建参数、如何后处理、forward 用哪个 kernel。
```

典型组合：

```text
load_format=safetensors + quantization=gptq
  → DefaultModelLoader 读 safetensors；
  → AutoGPTQConfig 解释 qweight / scales / qzeros。

load_format=bitsandbytes + quantization=bitsandbytes
  → BitsAndBytes loader + BitsAndBytesConfig 共同参与。
```

权重加载细节见：`03_weight_loading_and_param_mapping.md`。

---

## 11. kv_cache_dtype 是另一条配置线

`kv_cache_dtype` 最终进入：

```text
CacheConfig.cache_dtype
```

位置：`code/vllm/vllm/config/cache.py:75`

它控制的是 KV cache 存储格式，不等同于 `ModelConfig.quantization`。

```text
quantization：
  控制权重 / activation / MoE / layer quant method。

kv_cache_dtype：
  控制推理时历史 K/V cache 的 dtype / quant mode。
```

`CacheConfig` 还包含：

```text
calculate_kv_scales
kv_cache_dtype_skip_layers
```

位置：`code/vllm/vllm/config/cache.py:110`

KV cache 量化的完整机制见：`07_kv_cache_quantization.md`。

---

## 12. 几种典型配置路径

### 12.1 不启用权重量化

```text
用户不传 --quantization
checkpoint 没有 quantization_config
  → ModelConfig.quantization = None
  → VllmConfig.quant_config = None
```

结果：layer 使用未量化 method。

### 12.2 checkpoint 自带量化配置

```text
config.json.quantization_config.quant_method = "awq"
  → ModelConfig._verify_quantization() 推断 quantization
  → get_quant_config() 读取 HF quantization_config
  → AutoAWQConfig.from_config(...)
  → VllmConfig.quant_config
```

### 12.3 用户显式指定 checkpoint 量化方法

```text
--quantization gptq
  → ModelConfig.quantization = "gptq"
  → 如果 checkpoint 标注不匹配则报错
  → get_quant_config() 读取 GPTQ 配置
  → AutoGPTQConfig
```

### 12.4 用户启用在线量化

```text
--quantization fp8_per_tensor
  → resolve_quantization_config()
  → QuantizationConfigArgs
  → get_quant_config()
  → OnlineQuantizationConfig(args)
```

### 12.5 用户只量化 KV cache

```text
--kv-cache-dtype fp8
  → CacheConfig.cache_dtype = fp8
  → ModelConfig.quantization 可以仍然是 None
```

这说明 KV cache 可以独立量化，不要求权重量化。

---

## 13. 调试时看哪些配置位置

```text
用户参数是否进入：
  vllm/vllm/engine/arg_utils.py:748
  vllm/vllm/engine/arg_utils.py:1617
  vllm/vllm/engine/arg_utils.py:1832

方法名为什么变成某个值：
  vllm/vllm/config/model.py:970
  vllm/vllm/config/model.py:1014
  vllm/vllm/config/model.py:1038

最终 QuantizationConfig 对象是什么：
  vllm/vllm/config/vllm.py:609
  vllm/vllm/model_executor/model_loader/weight_utils.py:240
  vllm/vllm/model_executor/layers/quantization/__init__.py:108

KV cache dtype 为什么是某个值：
  vllm/vllm/config/cache.py:75
  vllm/vllm/engine/arg_utils.py:1832
  vllm/vllm/utils/torch_utils.py:373
```

---

## 14. 容易混淆的点

### 14.1 `ModelConfig.quantization` 是 QuantizationConfig 吗？

不是。它只是方法名字符串。

真正对象是：

```text
VllmConfig.quant_config
```

### 14.2 `quantization_config` 是否总来自用户？

不是。

```text
HF config 的 quantization_config：
  checkpoint 自带量化元数据。

ModelConfig.quantization_config：
  用户侧在线量化配置。
```

### 14.3 `load_format` 是否等于量化方法？

不是。`load_format` 决定怎么读文件，`quantization` 决定怎么解释和执行权重。

### 14.4 `kv_cache_dtype=auto` 是否一定不量化 KV cache？

不一定。`auto` 可能从模型量化元数据推断出 FP8 / NVFP4 等 KV cache dtype。

---

## 15. 最小心智模型

```text
EngineArgs：
  收集用户输入。

ModelConfig：
  确定量化方法名，并和 checkpoint metadata 对齐。

VllmConfig：
  构造真正的 QuantizationConfig 对象，并校验平台 / dtype。

QuantizationConfig：
  后续供模型 layer 查询 quant_method。
```

一句话总结：

```text
02 这篇只解释“配置如何变成 QuantizationConfig”；至于 QuantizationConfig 如何创建参数、加载权重和调用 kernel，交给后续 layer / loader / backend 专题展开。
```
