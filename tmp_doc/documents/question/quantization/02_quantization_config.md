# 02. 用户量化配置如何变成 QuantizationConfig？

源码位置：

- `vllm/vllm/engine/arg_utils.py`
- `vllm/vllm/config/model.py`
- `vllm/vllm/config/quantization.py`
- `vllm/vllm/config/vllm.py`
- `vllm/vllm/config/cache.py`
- `vllm/vllm/config/load.py`
- `vllm/vllm/model_executor/model_loader/weight_utils.py`
- `vllm/vllm/model_executor/model_loader/utils.py`
- `vllm/vllm/model_executor/layers/quantization/`
- `vllm/vllm/model_executor/layers/linear.py`
- `vllm/vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/vllm/platforms/interface.py`

本问题关注：用户传入的 `--quantization`、`--quantization-config`、模型 `config.json` 里的 `quantization_config`、额外量化配置文件、`load_format`、`kv_cache_dtype`，最终如何变成 vLLM 内部的 `QuantizationConfig` 对象，并进一步影响 layer 权重创建、权重加载、后处理和 kernel 选择。

---

## 1. 一句话回答

用户量化配置会先进入 `EngineArgs` / `ModelConfig`，在 `ModelConfig._verify_quantization()` 阶段确定“量化方法名”，再由 `VllmConfig._get_quantization_config()` 调用 `get_quant_config()` 把方法名和 checkpoint 配置解析成具体的 `QuantizationConfig` 子类实例，最后在模型初始化时传给 Linear / MoE / Attention layer，变成每个 layer 的 `quant_method`。

主链路是：

```text
CLI / EngineArgs
  → EngineArgs.__post_init__()
  → resolve_quantization_config()
  → ModelConfig.quantization / ModelConfig.quantization_config
  → ModelConfig._verify_quantization()
  → VllmConfig._get_quantization_config()
  → get_quant_config()
  → get_quantization_config(method).from_config(...)
  → VllmConfig.quant_config
  → initialize_model()
  → model layers receive quant_config
  → layer.quant_method
  → create_weights() / apply() / process_weights_after_loading()
```

所以：

```text
ModelConfig.quantization 是“量化方法名”；
VllmConfig.quant_config 是“解析后的量化配置对象”；
layer.quant_method 是“具体执行这个 layer 的量化逻辑”。
```

---

## 2. 最小主链路

可以先只记住这条线：

```text
用户参数 / checkpoint metadata
  → 解析出 quantization 方法名
  → 校验平台和 dtype 是否支持
  → 构造 QuantizationConfig 子类
  → 每个可量化 layer 向 quant_config 询问 get_quant_method(layer, prefix)
  → quant_method 创建权重参数并执行量化 kernel
```

典型对象关系是：

```text
EngineArgs.quantization
EngineArgs.quantization_config
        ↓
ModelConfig.quantization
ModelConfig.quantization_config
        ↓
VllmConfig.quant_config
        ↓
QuantizationConfig 子类
        ↓
QuantizeMethodBase / LinearMethodBase / FusedMoEMethodBase / KVCacheMethod
```

这里最容易混淆的是：

```text
quantization：权重 / layer 量化方法名，例如 awq、gptq、fp8、compressed-tensors、online。
quantization_config：用户侧在线量化配置，或 checkpoint 里的 HF 量化元数据。
kv_cache_dtype：KV Cache 存储 dtype，不等同于权重量化方法。
load_format：权重文件读取方式，不等同于量化方法。
```

---

## 3. 用户配置从哪里进入

### 3.1 EngineArgs 字段

入口在 `engine/arg_utils.py`。

相关字段：

```text
load_format
kv_cache_dtype
quantization
quantization_config
calculate_kv_scales
kv_cache_dtype_skip_layers
```

位置：`engine/arg_utils.py:437`、`engine/arg_utils.py:540`、`engine/arg_utils.py:680`

其中：

```text
--quantization / -q
  → EngineArgs.quantization

--quantization-config
  → EngineArgs.quantization_config

--kv-cache-dtype
  → EngineArgs.kv_cache_dtype

--load-format
  → EngineArgs.load_format
```

### 3.2 EngineArgs 先解析在线量化配置

`EngineArgs.__post_init__()` 会调用：

```python
self.quantization_config = resolve_quantization_config(
    self.quantization, self.quantization_config
)
```

位置：`engine/arg_utils.py:748` 到 `engine/arg_utils.py:752`

这一步只处理用户侧 `--quantization-config` 和在线量化 shorthand，不会读取 checkpoint 权重。

### 3.3 EngineArgs 创建 ModelConfig / CacheConfig / LoadConfig

创建 `ModelConfig` 时会把：

```text
quantization
quantization_config
allow_deprecated_quantization
```

传进去。

位置：`engine/arg_utils.py:1617` 到 `engine/arg_utils.py:1619`

创建 `CacheConfig` 前，会先解析：

```python
resolved_cache_dtype = resolve_kv_cache_dtype_string(
    self.kv_cache_dtype, model_config
)
```

位置：`engine/arg_utils.py:1832` 到 `engine/arg_utils.py:1835`

这说明权重量化和 KV cache dtype 在配置阶段已经分成两条线。

---

## 4. ModelConfig 里保存了什么

核心字段在 `config/model.py`：

```python
quantization: QuantizationMethods | str | None = None
quantization_config: dict[str, Any] | QuantizationConfigArgs | None = None
allow_deprecated_quantization: bool = False
```

位置：`config/model.py:197` 到 `config/model.py:207`

含义是：

```text
quantization：
  量化方法名。
  如果用户没有指定，vLLM 会尝试从模型 config.json 的 quantization_config 推断。

quantization_config：
  用户侧在线量化配置。
  它携带 linear / moe / ignore 等规则。

allow_deprecated_quantization：
  是否允许已废弃的量化方法。
```

`quantization` 还会被统一转成小写：

```python
@field_validator("quantization", mode="before")
def validate_quantization_before(cls, value):
    if isinstance(value, str):
        return value.lower()
```

位置：`config/model.py:749` 到 `config/model.py:754`

所以用户写 `AWQ`、`awq`，后续看到的都是小写方法名。

---

## 5. `--quantization-config` 解决什么问题

`--quantization-config` 的解析在 `config/quantization.py`。

核心对象是：

```text
QuantSpec
  weight
  activation

QuantizationConfigArgs
  linear
  moe
  ignore
```

位置：`config/quantization.py:63` 到 `config/quantization.py:93`

它主要服务在线量化：

```text
linear：应用到 LinearBase 层的量化规则；
moe：应用到 FusedMoE / RoutedExperts 的量化规则；
ignore：按 layer 名跳过量化。
```

支持的用户侧量化 key 包括：

```text
fp8_per_tensor_static
fp8_per_tensor_dynamic
fp8_per_token
fp8_per_channel_static
fp8_per_block_static
fp8_per_block_dynamic
mxfp8
mxfp4
int8_per_channel_static
```

位置：`config/quantization.py:23` 到 `config/quantization.py:34`

---

## 6. 在线量化 shorthand 如何展开

`config/quantization.py` 里定义了 `--quantization` 可以直接使用的在线量化 shorthand：

```text
fp8_per_tensor
fp8_per_block
fp8_per_channel
mxfp8
int8_per_channel_weight_only
online
```

位置：`config/quantization.py:112` 到 `config/quantization.py:144`

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

而：

```text
--quantization int8_per_channel_weight_only
```

只给 MoE 配置 int8 weight-only：

```text
QuantizationConfigArgs(
  moe=QuantSpec(weight=kInt8StaticChannelSym),
)
```

这意味着：

```text
在线量化 shorthand 本质上不是 checkpoint 格式，
而是把 fp16/bf16 权重在加载过程中转成某种量化形式。
```

---

## 7. resolve_quantization_config 的规则

入口：`resolve_quantization_config()`

位置：`config/quantization.py:147` 到 `config/quantization.py:183`

它的规则是：

```text
1. 如果 quantization 不是在线 shorthand：
     - 不允许再传 quantization_config；
     - 返回 None。

2. 如果 quantization 是在线 shorthand：
     - 先生成 base QuantizationConfigArgs。

3. 如果用户还传了 quantization_config：
     - 用户显式字段覆盖 shorthand；
     - 未显式指定的字段继承 shorthand。

4. 如果 quantization == "online"：
     - 必须依赖 quantization_config 提供 linear / moe 规则。
```

可以理解成：

```text
--quantization fp8_per_tensor 是快捷写法；
--quantization online --quantization-config {...} 是完整写法；
两者合用时，quantization_config 的字段优先。
```

---

## 8. checkpoint 自带量化配置如何识别

用户没有显式传 `--quantization` 时，vLLM 会从模型配置里推断。

入口是：

```python
ModelConfig._verify_quantization()
```

位置：`config/model.py:970`

它先拿：

```python
quant_cfg = self.model_arch_config.quantization_config
```

位置：`config/model.py:975` 到 `config/model.py:979`

如果 checkpoint 的 `config.json` 里有：

```json
{
  "quantization_config": {
    "quant_method": "awq"
  }
}
```

那么 `quant_method` 会成为候选量化方法。

如果用户没有传 `--quantization`：

```text
self.quantization = quant_method
```

如果用户显式传了 `--quantization`，则必须和 checkpoint 里的方法一致，否则报错。

位置：`config/model.py:1038` 到 `config/model.py:1048`

---

## 9. override_quantization_method 为什么存在

并不是所有 checkpoint 的 `quant_method` 都能直接映射到 vLLM 内部方法。

例如：

```text
GPTQ checkpoint 可能更适合走 gptq_marlin；
AWQ checkpoint 可能更适合走 awq_marlin；
某些 ModelOpt / MXFP4 / DeepSeek V4 FP8 checkpoint 需要特殊识别；
用户指定 humming 时要优先走 humming。
```

所以 `_verify_quantization()` 会按顺序遍历量化方法，并调用：

```python
method.override_quantization_method(
    quant_cfg, self.quantization, hf_config=self.hf_config
)
```

位置：`config/model.py:1014` 到 `config/model.py:1019`

`override_quantization_method()` 的抽象定义在：

```python
QuantizationConfig.override_quantization_method(...)
```

位置：`model_executor/layers/quantization/base_config.py:119` 到 `model_executor/layers/quantization/base_config.py:137`

它的意义是：

```text
checkpoint 声明的是“存储格式”；
vLLM 最终选择的是“运行时量化后端”。
```

---

## 10. 量化方法名如何校验

### 10.1 是否是 vLLM 已知方法

`_verify_quantization()` 会检查：

```python
if self.quantization not in supported_quantization:
    raise ValueError(...)
```

位置：`config/model.py:1050` 到 `config/model.py:1055`

`supported_quantization` 来自：

```python
me_quant.QUANTIZATION_METHODS
```

方法列表定义在 `model_executor/layers/quantization/__init__.py:12` 到 `model_executor/layers/quantization/__init__.py:47`。

常见方法包括：

```text
awq
fp8
auto_gptq / gptq / gptq_marlin
awq_marlin
compressed-tensors
bitsandbytes
torchao
modelopt / modelopt_fp4 / modelopt_mxfp8 / modelopt_mixed
mxfp4
deepseek_v4_fp8
online
fp8_per_tensor / fp8_per_block / fp8_per_channel / mxfp8
```

### 10.2 当前平台是否支持

随后会调用：

```python
current_platform.verify_quantization(self.quantization)
```

位置：`config/model.py:1056`

平台抽象里默认逻辑是：

```python
if cls.supported_quantization and quant not in cls.supported_quantization:
    raise ValueError(...)
```

位置：`platforms/interface.py:824` 到 `platforms/interface.py:831`

这说明：

```text
同一个 quantization 方法是否可用，取决于当前 Platform 的 supported_quantization。
```

### 10.3 是否废弃

废弃方法在：

```python
DEPRECATED_QUANTIZATION_METHODS = ["fbgemm_fp8", "fp_quant"]
```

位置：`model_executor/layers/quantization/__init__.py:49` 到 `model_executor/layers/quantization/__init__.py:52`

如果没有传 `--allow-deprecated-quantization`，会直接报错。

位置：`config/model.py:1058` 到 `config/model.py:1071`

---

## 11. VllmConfig 什么时候生成真正的 QuantizationConfig

`ModelConfig` 里只有方法名和用户参数，真正的 `QuantizationConfig` 对象在 `VllmConfig` 里生成。

字段：

```python
quant_config: QuantizationConfig | None = None
```

位置：`config/vllm.py:334`

`VllmConfig.__post_init__()` 中：

```python
if self.quant_config is None and self.model_config is not None:
    self.quant_config = VllmConfig._get_quantization_config(
        self.model_config, self.load_config
    )
```

位置：`config/vllm.py:910` 到 `config/vllm.py:913`

这一步之后，`VllmConfig` 才真正携带：

```text
具体 QuantizationConfig 子类实例
```

例如：

```text
Fp8Config
AutoAWQConfig
AutoGPTQConfig
CompressedTensorsConfig
BitsAndBytesConfig
OnlineQuantizationConfig
ModelOptNvFp4Config
Mxfp4Config
```

---

## 12. _get_quantization_config 做了哪些校验

入口：`VllmConfig._get_quantization_config()`

位置：`config/vllm.py:611` 到 `config/vllm.py:642`

流程是：

```text
1. 如果 model_config.quantization is None：
     返回 None。

2. 调用 get_quant_config(model_config, load_config)：
     读取 checkpoint / 用户配置并构造 QuantizationConfig。

3. 获取当前设备 capability：
     current_platform.get_device_capability()

4. 校验 GPU capability >= quant_config.get_min_capability()。

5. 校验 model_config.dtype 在 quant_config.get_supported_act_dtypes() 中。

6. 调用 quant_config.maybe_update_config(model_name, hf_config)。

7. 返回 quant_config。
```

这一步做的是运行时可用性校验：

```text
方法名存在，不代表当前 GPU / dtype / checkpoint 一定能跑。
```

例如 `Fp8Config` 要求：

```text
get_min_capability() = 75
get_supported_act_dtypes() = [torch.bfloat16, torch.half]
```

位置：`model_executor/layers/quantization/fp8.py:139` 到 `model_executor/layers/quantization/fp8.py:149`

---

## 13. get_quant_config 如何读取配置

真正构造 `QuantizationConfig` 的入口是：

```python
def get_quant_config(model_config, load_config) -> QuantizationConfig:
```

位置：`model_executor/model_loader/weight_utils.py:240`

### 13.1 先用方法名找到配置类

```python
quant_cls = get_quantization_config(model_config.quantization)
```

位置：`model_executor/model_loader/weight_utils.py:245`

`get_quantization_config()` 在注册表里把方法名映射到类。

位置：`model_executor/layers/quantization/__init__.py:108` 到 `model_executor/layers/quantization/__init__.py:182`

例如：

```text
"awq" / "awq_marlin" / "auto_awq" → AutoAWQConfig
"gptq" / "gptq_marlin" / "auto_gptq" → AutoGPTQConfig
"fp8" → Fp8Config
"compressed-tensors" → CompressedTensorsConfig
"bitsandbytes" → BitsAndBytesConfig
"online" → OnlineQuantizationConfig
```

### 13.2 优先读 HF config 里的量化配置

`get_quant_config()` 优先看：

```text
model_config.hf_config.quantization_config
model_config.hf_config.text_config.quantization_config
model_config.hf_config.compression_config
```

位置：`model_executor/model_loader/weight_utils.py:247` 到 `model_executor/model_loader/weight_utils.py:255`

如果存在，就调用：

```python
return quant_cls.from_config(hf_quant_config)
```

位置：`model_executor/model_loader/weight_utils.py:275` 到 `model_executor/model_loader/weight_utils.py:291`

这就是大多数量化 checkpoint 的路径。

### 13.3 再看 hf_overrides

如果 HF config 里没有量化配置，会尝试从 `hf_overrides` 读取：

```text
quantization_config_file
quantization_config_dict_json
```

位置：`model_executor/model_loader/weight_utils.py:293` 到 `model_executor/model_loader/weight_utils.py:320`

这给了用户手动补量化配置的入口。

### 13.4 在线量化不读 checkpoint config

如果 `model_config.quantization_config` 不为空，会直接构造：

```python
OnlineQuantizationConfig(args=model_config.quantization_config)
```

位置：`model_executor/model_loader/weight_utils.py:322` 到 `model_executor/model_loader/weight_utils.py:331`

注释也说明：

```text
Online quantization doesn't read from checkpoint configs - it quantizes fp16/bf16 weights on the fly during loading.
```

### 13.5 bitsandbytes 是特殊 inflight 路径

如果量化方法是：

```text
bitsandbytes
```

会走：

```python
return quant_cls.from_config({})
```

位置：`model_executor/model_loader/weight_utils.py:333` 到 `model_executor/model_loader/weight_utils.py:335`

### 13.6 最后才查找量化配置文件

如果前面都没有拿到，会下载 / 定位模型目录，然后查：

```python
possible_config_filenames = quant_cls.get_config_filenames()
```

位置：`model_executor/model_loader/weight_utils.py:336` 到 `model_executor/model_loader/weight_utils.py:360`

例如 GPTQ：

```python
get_config_filenames() → ["quantize_config.json"]
```

位置：`model_executor/layers/quantization/auto_gptq.py:190` 到 `auto_gptq.py:193`

如果类没有配置文件名，则直接：

```python
return quant_cls()
```

位置：`model_executor/model_loader/weight_utils.py:360` 到 `model_executor/model_loader/weight_utils.py:364`

---

## 14. QuantizationConfig 抽象负责什么

抽象类在：`model_executor/layers/quantization/base_config.py:77`

它定义了量化配置必须提供的能力。

### 14.1 方法名

```python
def get_name(self) -> QuantizationMethods
```

表示这个配置对应哪种量化方法。

### 14.2 支持的 activation dtype

```python
def get_supported_act_dtypes(self) -> list[torch.dtype]
```

用于校验 `model_config.dtype`。

### 14.3 最低 GPU capability

```python
def get_min_capability(cls) -> int
```

用于校验当前 GPU 是否支持对应 kernel。

### 14.4 配置文件名

```python
def get_config_filenames() -> list[str]
```

用于从模型目录中查找额外量化配置文件。

### 14.5 从配置构造对象

```python
def from_config(cls, config: dict[str, Any]) -> QuantizationConfig
```

把 checkpoint 的量化 JSON 变成 Python 对象。

### 14.6 为具体 layer 选择 quant method

```python
def get_quant_method(self, layer, prefix) -> QuantizeMethodBase | None
```

这是最重要的方法。

它回答：

```text
这个 layer 是否支持当前量化方法？
如果支持，应该使用哪个 QuantizeMethodBase？
如果不支持，返回 None 或跳过。
```

位置：`model_executor/layers/quantization/base_config.py:157` 到 `base_config.py:170`

---

## 15. 量化配置如何进入模型初始化

模型初始化入口：

```python
initialize_model(vllm_config)
```

位置：`model_executor/model_loader/utils.py:40`

如果 `vllm_config.quant_config` 不为空，会先调用：

```python
configure_quant_config(vllm_config.quant_config, model_class)
```

位置：`model_executor/model_loader/utils.py:54` 到 `model_loader/utils.py:55`

然后模型类通常以新风格接收：

```python
model_class(vllm_config=vllm_config, prefix=prefix)
```

位置：`model_executor/model_loader/utils.py:59` 到 `model_loader/utils.py:63`

对于旧风格模型，vLLM 会尝试把：

```text
quant_config=vllm_config.quant_config
```

作为参数传入。

位置：`model_executor/model_loader/utils.py:87` 到 `model_loader/utils.py:88`

所以模型构造阶段能拿到的是：

```text
VllmConfig.quant_config
```

而不是原始 CLI 字符串。

---

## 16. Linear layer 如何使用 quant_config

核心在：`model_executor/layers/linear.py`

`LinearBase.__init__()` 接收：

```python
quant_config: QuantizationConfig | None = None
prefix: str = ""
```

位置：`model_executor/layers/linear.py:242` 到 `linear.py:250`

然后决定 `self.quant_method`：

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
else:
    raise ValueError("All linear layers should support quant method.")
```

位置：`model_executor/layers/linear.py:268` 到 `linear.py:274`

这说明：

```text
LinearBase 不知道自己应该怎么量化；
它只把自己和 prefix 交给 QuantizationConfig；
QuantizationConfig 决定返回哪个 QuantizeMethodBase。
```

之后：

```python
self.quant_method.create_weights(...)
```

位置：`model_executor/layers/linear.py:338` 到 `linear.py:343`

forward 时：

```python
output = self.quant_method.apply(self, x, bias)
```

位置：`model_executor/layers/linear.py:378`

所以量化最终影响两件事：

```text
1. 这个 layer 注册哪些参数；
2. forward 用哪个 kernel / 算法执行矩阵乘。
```

---

## 17. MoE layer 如何使用 quant_config

MoE 入口是工厂函数：

```python
def FusedMoE(..., quant_config: QuantizationConfig | None = None, prefix: str = "", ...)
```

位置：`model_executor/layers/fused_moe/layer.py:102` 到 `layer.py:117`

它会创建完整 MoE 执行管线：

```text
Router
RoutedExperts
MoERunner
```

位置：`model_executor/layers/fused_moe/layer.py:148` 到 `layer.py:154`

对量化来说，关键是：

```text
RoutedExperts / FusedMoE 会向 quant_config 请求适合 MoE 的 quant method。
```

例如 `OnlineQuantizationConfig.get_quant_method()` 中：

```text
LinearBase → _ONLINE_LINEAR_METHODS
RoutedExperts → _ONLINE_MOE_METHODS
```

位置：`model_executor/layers/quantization/online/base.py:145` 到 `online/base.py:170`

因此同一个 `QuantizationConfig` 可以对：

```text
Linear 层
MoE experts
Attention KV cache
```

返回不同的量化 method。

---

## 18. Attention / KV cache 也可能通过 quant_config 接入

`QuantizationConfig.get_quant_method()` 不只服务 Linear。

以 FP8 为例：

```python
elif isinstance(layer, Attention):
    return Fp8KVCacheMethod(self)
```

位置：`model_executor/layers/quantization/fp8.py:219` 到 `fp8.py:220`

这意味着：

```text
某些权重量化配置也可能携带 KV cache scale 映射或 Attention 侧量化方法。
```

同时 `QuantizationConfig` 还有：

```python
def get_cache_scale_mapper(self) -> WeightsMapper | None
```

位置：`model_executor/layers/quantization/base_config.py:172` 到 `base_config.py:179`

例如 `Fp8Config` 会把 checkpoint 中的 KV-cache scale 名字映射到 vLLM 模型里的名字。

位置：`model_executor/layers/quantization/fp8.py:223` 到 `fp8.py:234`

这解释了为什么量化配置不仅影响 Linear 权重，还可能影响：

```text
k_scale
v_scale
q_scale
attention prob scale
```

这些额外参数的加载。

---

## 19. 权重加载后为什么还要 process_weights_after_loading

有些量化方法不能只靠 `create_weights()` 完成。

模型权重加载后，vLLM 会遍历所有 module：

```python
quant_method = getattr(module, "quant_method", None)
if isinstance(quant_method, QuantizeMethodBase):
    quant_method.process_weights_after_loading(module)
```

位置：`model_executor/model_loader/utils.py:100` 到 `model_loader/utils.py:115`

这一步用于：

```text
- repack 权重；
- 转置权重；
- 在线量化 fp16/bf16 权重；
- 准备 kernel 需要的 layout；
- CPU offload 场景下临时把参数搬到目标设备处理。
```

因此量化完整生命周期是：

```text
QuantizationConfig.from_config()
  → layer.quant_method
  → quant_method.create_weights()
  → checkpoint 权重加载
  → quant_method.process_weights_after_loading()
  → quant_method.apply()
```

---

## 20. load_format 和 quantization 的边界

`LoadConfig.load_format` 在：`config/load.py:28`

它控制的是“用什么 loader / 文件格式读权重”。

常见值：

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

位置：`config/load.py:28` 到 `config/load.py:57`

它不是量化方法本身。

例如：

```text
--load-format safetensors
  只说明权重从 safetensors 读；
  不说明权重是不是 AWQ / GPTQ / FP8。

--quantization awq
  说明用 AWQ 量化配置和 kernel；
  但权重文件仍可能是 safetensors / bin / pt。

--load-format bitsandbytes
  是特殊 loader 路径；
  同时 bitsandbytes 也有自己的 quantization 方法处理。
```

`DefaultModelLoader._prepare_weights()` 会根据 `load_format` 决定文件 pattern：

```text
hf → *.safetensors / *.bin / *.pt
safetensors → *.safetensors
mistral → consolidated*.safetensors
pt → *.pt
npcache → *.bin
```

位置：`model_executor/model_loader/default_loader.py:144` 到 `default_loader.py:184`

所以：

```text
load_format 决定“怎么读文件”；
quantization 决定“读出来的参数如何解释、如何创建 layer、用什么 kernel”。
```

---

## 21. kv_cache_dtype 是另一条配置线

KV cache dtype 在 `CacheConfig` 中：

```python
cache_dtype: CacheDType = "auto"
```

位置：`config/cache.py:75`

可选值包括：

```text
auto
float16
bfloat16
fp8
fp8_e4m3
fp8_e5m2
fp8_inc
fp8_ds_mla
turboquant_k8v4
turboquant_4bit_nc
turboquant_k3v4_nc
turboquant_3bit_nc
int8_per_token_head
fp8_per_token_head
nvfp4
```

位置：`config/cache.py:19` 到 `config/cache.py:35`

### 21.1 auto 会尝试从模型量化配置推断

`EngineArgs.create_engine_config()` 会调用：

```python
resolve_kv_cache_dtype_string(self.kv_cache_dtype, model_config)
```

位置：`engine/arg_utils.py:1832` 到 `engine/arg_utils.py:1835`

如果用户传的不是 `auto`，直接使用用户值。

如果是 `auto`，则读取：

```text
model_config.hf_config.quantization_config
```

并尝试从其中的 KV cache quant algo 推断。

位置：`utils/torch_utils.py:373` 到 `torch_utils.py:390`

ModelOpt 到 vLLM 的映射是：

```text
fp8 → fp8_e4m3
nvfp4 → nvfp4
```

位置：`utils/torch_utils.py:63` 到 `torch_utils.py:66`

### 21.2 什么算量化 KV cache

工具函数：

```python
def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return (
        kv_cache_dtype.startswith("fp8")
        or kv_cache_dtype.endswith("per_token_head")
        or kv_cache_dtype == "nvfp4"
    )
```

位置：`utils/torch_utils.py:74` 到 `torch_utils.py:79`

也就是说：

```text
KV cache 量化可以独立于权重量化存在。
```

### 21.3 calculate_kv_scales 和 skip layers

`CacheConfig` 还有：

```text
calculate_kv_scales
kv_cache_dtype_skip_layers
```

位置：`config/cache.py:110` 到 `config/cache.py:117`

含义是：

```text
calculate_kv_scales：
  已废弃；曾用于 fp8 KV cache 下动态计算 k_scale / v_scale。
  如果 False，优先从 checkpoint 加载 scale；没有则默认 1.0。

kv_cache_dtype_skip_layers：
  指定某些层跳过 KV cache 量化。
  可以按 layer index 或 attention type 名称跳过。
```

所以：

```text
kv_cache_dtype 控制 KV Cache 存储；
QuantizationConfig 控制权重 / layer 方法；
两者可能互相读取 metadata，但不是同一个配置。
```

---

## 22. 在线量化和 checkpoint 量化的区别

### 22.1 checkpoint 量化

例如 AWQ / GPTQ / compressed-tensors / ModelOpt：

```text
checkpoint 已经保存量化权重或 scale；
vLLM 读取 quantization_config；
构造对应 QuantizationConfig；
layer 注册 qweight / qzeros / scales 等参数；
加载 checkpoint 中的量化参数；
必要时 repack；
forward 使用对应 kernel。
```

典型链路：

```text
config.json.quantization_config
  → ModelConfig._verify_quantization()
  → get_quant_config()
  → AutoAWQConfig.from_config() / AutoGPTQConfig.from_config() / ...
  → layer.quant_method
```

### 22.2 在线量化

例如：

```text
--quantization fp8_per_tensor
--quantization fp8_per_block
--quantization online --quantization-config {...}
```

链路是：

```text
EngineArgs.__post_init__()
  → resolve_quantization_config()
  → QuantizationConfigArgs
  → get_quant_config()
  → OnlineQuantizationConfig(args)
  → layer.quant_method
  → process_weights_after_loading() 中把 fp16/bf16 权重量化
```

`OnlineQuantizationConfig` 明确要求：

```text
至少 linear 或 moe 其中一个 spec 不为空。
```

位置：`model_executor/layers/quantization/online/base.py:78` 到 `online/base.py:89`

它不支持从 checkpoint config 里 `from_config()`：

```text
Use quantization_config or quantization='fp8_per_tensor'/'fp8_per_block' instead.
```

位置：`model_executor/layers/quantization/online/base.py:110` 到 `online/base.py:116`

---

## 23. 以 FP8 为例看落地过程

`Fp8Config` 在：`model_executor/layers/quantization/fp8.py:99`

它保存：

```text
is_checkpoint_fp8_serialized
activation_scheme
ignored_layers
weight_block_size
store_dtype
use_deep_gemm
```

位置：`model_executor/layers/quantization/fp8.py:102` 到 `fp8.py:138`

`from_config()` 会从 checkpoint 量化配置中读取：

```text
quant_method
activation_scheme
ignored_layers / modules_to_not_convert
weight_block_size
store_dtype
```

位置：`model_executor/layers/quantization/fp8.py:160` 到 `fp8.py:177`

当 layer 是 `LinearBase` 时：

```text
1. 如果 prefix 在 ignored_layers 中：返回 UnquantizedLinearMethod；
2. 如果 checkpoint 不是 fp8 serialized：返回在线 FP8 method；
3. 如果 checkpoint 已经是 fp8 serialized：返回 Fp8LinearMethod。
```

位置：`model_executor/layers/quantization/fp8.py:179` 到 `fp8.py:200`

当 layer 是 `RoutedExperts` 时：

```text
返回 Fp8MoEMethod / Fp8OnlineMoEMethod / Mxfp4MoEMethod。
```

位置：`model_executor/layers/quantization/fp8.py:201` 到 `fp8.py:218`

当 layer 是 `Attention` 时：

```text
返回 Fp8KVCacheMethod。
```

位置：`model_executor/layers/quantization/fp8.py:219` 到 `fp8.py:220`

这说明 FP8 不是单一 kernel 开关，而是一套：

```text
Linear + MoE + Attention KV scale + ignored layer + checkpoint/online 分支
```

共同组成的量化配置。

---

## 24. 以 AWQ / GPTQ 为例看配置读取

### 24.1 AWQ

`AutoAWQConfig` 在：`model_executor/layers/quantization/auto_awq.py:170`

它关心：

```text
weight_bits
group_size
zero_point
lm_head_quantized
modules_to_not_convert
full_config
```

位置：`model_executor/layers/quantization/auto_awq.py:182` 到 `auto_awq.py:199`

它会校验 AWQ 支持的 bit 数，目前核心是 4bit。

位置：`model_executor/layers/quantization/auto_awq.py:200` 到 `auto_awq.py:209`

### 24.2 GPTQ

`AutoGPTQConfig` 在：`model_executor/layers/quantization/auto_gptq.py:97`

它关心：

```text
bits
group_size
desc_act
sym
lm_head
dynamic
modules_in_block_to_quantize
```

位置：`model_executor/layers/quantization/auto_gptq.py:106` 到 `auto_gptq.py:116`

`from_config()` 从 checkpoint 量化配置里读取：

```text
bits
group_size
desc_act
sym
lm_head
modules_in_block_to_quantize
```

位置：`model_executor/layers/quantization/auto_gptq.py:195` 到 `auto_gptq.py:216`

GPTQ 还有 `dynamic` 配置，允许按模块正向匹配 / 负向匹配来覆盖或跳过量化。

位置：`model_executor/layers/quantization/auto_gptq.py:123` 到 `auto_gptq.py:145`

所以 AWQ / GPTQ 的 `QuantizationConfig` 不只是名字，还包含：

```text
bit width、group size、zero point、scale layout、跳过规则、动态 per-module override。
```

---

## 25. 常见路径拆解

### 25.1 用户不传 quantization，模型也没有 quantization_config

```text
ModelConfig.quantization = None
VllmConfig.quant_config = None
LinearBase.quant_method = UnquantizedLinearMethod
```

结果：

```text
按 model dtype 加载普通权重。
```

### 25.2 用户不传 quantization，但 checkpoint 有 quantization_config

```text
config.json.quantization_config.quant_method = "awq"
  → ModelConfig._verify_quantization() 推断 self.quantization = "awq"
  → get_quant_config() 读取 HF quantization_config
  → AutoAWQConfig.from_config(...)
  → Linear / MoE 使用 AWQ quant_method
```

### 25.3 用户传了 `--quantization gptq`

```text
EngineArgs.quantization = "gptq"
  → ModelConfig.quantization = "gptq"
  → 如果 checkpoint quant_method 不匹配则报错
  → get_quant_config() 读取 GPTQ config
  → AutoGPTQConfig
```

### 25.4 用户传了 `--quantization fp8_per_tensor`

```text
EngineArgs.__post_init__()
  → resolve_quantization_config()
  → QuantizationConfigArgs(linear=..., moe=...)
  → ModelConfig.quantization = "fp8_per_tensor"
  → get_quantization_config("fp8_per_tensor") 映射到 OnlineQuantizationConfig
  → get_quant_config() 返回 OnlineQuantizationConfig(args)
```

结果：

```text
加载 fp16/bf16 checkpoint，并在加载过程中在线量化。
```

### 25.5 用户只想量化 KV cache

例如：

```text
--kv-cache-dtype fp8
```

链路是：

```text
EngineArgs.kv_cache_dtype
  → resolve_kv_cache_dtype_string()
  → CacheConfig.cache_dtype
  → attention / KV cache 初始化逻辑
```

这不要求：

```text
ModelConfig.quantization != None
```

也就是说：

```text
KV cache 可以量化，但权重仍然不量化。
```

---

## 26. 容易混淆的点

### 26.1 `ModelConfig.quantization` 是不是 QuantizationConfig？

不是。

它只是字符串方法名：

```text
"awq" / "gptq" / "fp8" / "online" / ...
```

真正的对象是：

```text
VllmConfig.quant_config
```

### 26.2 `quantization_config` 是不是总来自用户？

不是。

有两种语境：

```text
ModelConfig.quantization_config：
  用户侧在线量化配置，类型可以是 QuantizationConfigArgs。

hf_config.quantization_config：
  checkpoint 自带量化元数据，通常来自 config.json。
```

名字相同，但语义不同。

### 26.3 `--load-format` 是不是量化方法？

不是。

它决定权重文件如何读取；量化方法决定权重如何解释和执行。

### 26.4 checkpoint 里的 `quant_method` 是否一定等于最终运行方法？

不一定。

`override_quantization_method()` 可能把 checkpoint format 映射成更合适的 vLLM 后端，例如 Marlin / ModelOpt / 特定模型专用后端。

### 26.5 `kv_cache_dtype=auto` 是否等于不量化 KV cache？

不一定。

`auto` 会尝试从模型量化配置里读取 KV cache quant algo；读不到才继续保持 auto，由下游按默认规则处理。

### 26.6 所有 layer 都一定会量化吗？

不一定。

原因包括：

```text
- ignored_layers / modules_to_not_convert；
- quantization_config.ignore；
- GPTQ dynamic 负向匹配；
- layer 类型不支持当前 quant method；
- MoE / Linear 分别有不同 spec；
- kv_cache_dtype_skip_layers 跳过某些 KV cache 层。
```

---

## 27. 调试时应该看哪些位置

如果想查用户参数是否进来了：

```text
engine/arg_utils.py:748
engine/arg_utils.py:1617
engine/arg_utils.py:1832
```

如果想查方法名为什么变成某个值：

```text
config/model.py:970
config/model.py:1014
config/model.py:1038
```

如果想查最终 QuantizationConfig 对象是什么：

```text
config/vllm.py:611
model_executor/model_loader/weight_utils.py:240
model_executor/layers/quantization/__init__.py:108
```

如果想查某个 layer 为什么量化 / 没量化：

```text
model_executor/layers/linear.py:268
model_executor/layers/quantization/<method>.py:get_quant_method()
model_executor/layers/quantization/online/base.py:145
```

如果想查权重加载后做了什么：

```text
model_executor/model_loader/utils.py:100
```

如果想查 KV cache dtype：

```text
config/cache.py:75
utils/torch_utils.py:373
engine/arg_utils.py:1832
```

---

## 28. 最小心智模型

可以把整套量化配置拆成三层：

```text
第一层：配置层
  CLI / EngineArgs / HF config
  决定 quantization 方法名、在线量化规则、KV cache dtype、load_format。

第二层：对象层
  VllmConfig.quant_config
  把方法名和 checkpoint metadata 解析成 QuantizationConfig 子类实例。

第三层：执行层
  layer.quant_method
  决定参数如何创建、权重如何加载后处理、forward 用哪个 kernel。
```

再压缩成一句话：

```text
quantization 先决定“用哪类量化”，QuantizationConfig 再决定“每个 layer 怎么量化”，quant_method 最终决定“权重怎么存、kernel 怎么跑”。
```
