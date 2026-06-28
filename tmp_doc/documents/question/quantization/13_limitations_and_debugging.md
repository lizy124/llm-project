# 13. 量化有哪些限制和调试入口？

源码位置：

- `vllm/vllm/config/model.py`
- `vllm/vllm/config/vllm.py`
- `vllm/vllm/config/quantization.py`
- `vllm/vllm/config/cache.py`
- `vllm/vllm/platforms/interface.py`
- `vllm/vllm/platforms/cuda.py`
- `vllm/vllm/platforms/rocm.py`
- `vllm/vllm/model_executor/layers/linear.py`
- `vllm/vllm/model_executor/layers/quantization/__init__.py`
- `vllm/vllm/model_executor/layers/quantization/base_config.py`
- `vllm/vllm/model_executor/layers/quantization/fp8.py`
- `vllm/vllm/model_executor/layers/quantization/auto_awq.py`
- `vllm/vllm/model_executor/layers/quantization/auto_gptq.py`
- `vllm/vllm/model_executor/layers/quantization/bitsandbytes.py`
- `vllm/vllm/model_executor/layers/quantization/online/base.py`
- `vllm/vllm/model_executor/kernels/linear/base.py`
- `vllm/vllm/model_executor/kernels/linear/scaled_mm/`
- `vllm/vllm/model_executor/kernels/linear/mixed_precision/`
- `vllm/vllm/model_executor/model_loader/weight_utils.py`
- `vllm/vllm/model_executor/model_loader/default_loader.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：vLLM 量化常见限制来自哪里、错误通常在哪一层抛出、kernel 为什么 fallback、checkpoint scale / zero point 为什么加载失败、KV cache quantization 为什么影响精度，以及调试时应该按什么路径逐层定位。

---

## 1. 一句话回答

vLLM 的量化问题通常不是单点 bug，而是下面几层约束没有同时满足：

```text
用户参数 / HF quantization_config
  → vLLM ModelConfig._verify_quantization()
  → VllmConfig._get_quantization_config()
  → QuantizationConfig.from_config()
  → QuantizationConfig.get_quant_method(layer, prefix)
  → Linear / MoE / Attention 创建量化参数
  → weight loader 加载 qweight / scales / zero point
  → process_weights_after_loading() 做 repack / padding / scale 转换
  → kernel selector 判断 is_supported() / can_implement()
  → forward 调 quant_method.apply()
  → custom op / Cutlass / Marlin / Triton / DeepGEMM / FlashInfer 执行
```

所以调试量化问题时，最小心智模型是：

```text
先确认“选了什么量化方法”，
再确认“checkpoint 是否匹配”，
再确认“每层实际用了什么 quant_method”，
再确认“kernel 为什么支持 / 不支持”，
最后再看 runtime 输出、精度和性能。
```

---

## 2. 量化限制来自哪几层

vLLM 量化限制大致可以分成七层。

```text
1. 配置层：
   quantization 名称、HF config、CLI 参数是否一致。

2. 平台层：
   当前平台是否支持该 quantization method。

3. dtype / capability 层：
   GPU compute capability 和 activation dtype 是否满足要求。

4. checkpoint 层：
   qweight / scales / qzeros / k_scale / v_scale 是否存在且命名可映射。

5. layer 层：
   当前 layer 类型是否有对应 quant_method。

6. kernel 层：
   当前 shape、group_size、scale 粒度、TP shard 是否满足 kernel 约束。

7. runtime 层：
   CUDA graph、torch.compile、KV cache dtype、sleep/wake、batch shape 是否兼容。
```

这也是调试顺序：

```text
配置 → 平台 → dtype/capability → checkpoint → layer → kernel → runtime
```

---

## 3. 配置层：量化方法如何被验证

### 3.1 支持的量化方法列表

入口：`quantization/__init__.py`

核心列表是：

```python
QuantizationMethods = Literal[
    "awq",
    "auto_awq",
    "fp8",
    "modelopt",
    "auto_gptq",
    "gptq",
    "gptq_marlin",
    "awq_marlin",
    "compressed-tensors",
    "bitsandbytes",
    "experts_int8",
    "quark",
    "torchao",
    "inc",
    "mxfp4",
    "deepseek_v4_fp8",
    "online",
    "fp8_per_tensor",
    "fp8_per_block",
    "fp8_per_channel",
    "int8_per_channel_weight_only",
    "mxfp8",
]
```

位置：`quantization/__init__.py:12` 到 `quantization/__init__.py:46`

如果用户传入未知方法，会在：

```python
def get_quantization_config(quantization: str)
```

里抛出：

```text
Invalid quantization method: xxx
```

位置：`quantization/__init__.py:108` 到 `quantization/__init__.py:110`

### 3.2 ModelConfig 如何校验 quantization

入口：

```python
ModelConfig._verify_quantization()
```

位置：`config/model.py:970`

它做几件事：

```text
1. 读取 HF config 里的 quantization_config；
2. 从 quantization_config["quant_method"] 推断 checkpoint 方法；
3. 允许某些后端 override，例如 GPTQ → auto_gptq；
4. 如果用户显式传了 --quantization，则必须和 checkpoint 推断结果一致；
5. 检查该方法是否在 QUANTIZATION_METHODS 中；
6. 调 current_platform.verify_quantization() 做平台白名单校验；
7. 检查 deprecated quantization 是否允许。
```

不一致时会报：

```text
Quantization method specified in the model config (...) does not match the quantization method specified in the `quantization` argument (...).
```

位置：`config/model.py:1042` 到 `config/model.py:1048`

未知方法会报：

```text
Unknown quantization method: xxx. Must be one of [...]
```

位置：`config/model.py:1050` 到 `config/model.py:1055`

### 3.3 deprecated quantization

`fbgemm_fp8` 和 `fp_quant` 在：

```text
DEPRECATED_QUANTIZATION_METHODS
```

位置：`quantization/__init__.py:49` 到 `quantization/__init__.py:52`

如果没设置允许项，会报：

```text
The quantization method %s is deprecated and will be removed in future versions of vLLM. To bypass, set `--allow-deprecated-quantization`.
```

位置：`config/model.py:1058` 到 `config/model.py:1071`

---

## 4. 平台层：current_platform.verify_quantization()

平台校验入口：

```python
current_platform.verify_quantization(self.quantization)
```

位置：`config/model.py:1056`

默认实现：

```python
if cls.supported_quantization and quant not in cls.supported_quantization:
    raise ValueError(
        f"{quant} quantization is currently not supported in {cls.device_name}."
    )
```

位置：`platforms/interface.py:824` 到 `platforms/interface.py:831`

这说明：

```text
同一个 quantization method，在 CUDA / ROCm / XPU / CPU 上可能支持矩阵不同。
```

例如 ROCm 显式列了自己的支持列表：

```text
awq / auto_awq / awq_marlin
gptq / gptq_marlin / auto_gptq
fp8 / deepseek_v4_fp8
compressed-tensors / quark / mxfp4 / mxfp8
torchao / bitsandbytes / modelopt / online 等
```

位置：`platforms/rocm.py:441` 到 `platforms/rocm.py:466`

因此调试平台问题时要先看：

```text
current_platform.device_name
current_platform.supported_quantization
current_platform.is_cuda() / is_rocm() / is_xpu() / is_cpu()
```

---

## 5. capability 和 activation dtype 限制

### 5.1 VllmConfig 的统一校验

入口：

```python
VllmConfig._get_quantization_config(model_config, load_config)
```

位置：`config/vllm.py:609`

它会：

```text
1. get_quant_config() 读取具体 QuantizationConfig；
2. 读取当前设备 capability；
3. 和 quant_config.get_min_capability() 比较；
4. 检查 model_config.dtype 是否在 quant_config.get_supported_act_dtypes() 中；
5. 调 quant_config.maybe_update_config()。
```

位置：`config/vllm.py:615` 到 `config/vllm.py:641`

GPU capability 不满足时：

```text
The quantization method xxx is not supported for the current GPU. Minimum capability: y. Current capability: z.
```

位置：`config/vllm.py:621` 到 `config/vllm.py:629`

dtype 不满足时：

```text
torch.xxx is not supported for quantization method xxx. Supported dtypes: [...]
```

位置：`config/vllm.py:630` 到 `config/vllm.py:636`

### 5.2 常见后端的 capability / dtype

#### FP8

```text
supported act dtype：bf16 / fp16
min capability：75
```

位置：`fp8.py:143` 到 `fp8.py:149`

注意：

```text
min capability=75 不等于所有 GPU 都有原生 FP8 GEMM；
低架构可能走 Marlin FP8 weight-only fallback；
真正 kernel 还要看后续 kernel selector。
```

#### AutoAWQ

```text
supported act dtype：fp16 / bf16
min capability：75
```

位置：`auto_awq.py:224` 到 `auto_awq.py:230`

#### AutoGPTQ

```text
supported act dtype：fp16 / bf16
min capability：60
```

位置：`auto_gptq.py:182` 到 `auto_gptq.py:188`

#### BitsAndBytes

```text
supported act dtype：fp32 / fp16 / bf16
min capability：70
```

位置：`bitsandbytes.py:99` 到 `bitsandbytes.py:105`

#### Online quantization

```text
supported act dtype：bf16 / fp16
min capability：75
```

位置：`online/base.py:96` 到 `online/base.py:104`

---

## 6. checkpoint 配置如何加载

量化配置真正加载入口：

```python
get_quant_config(model_config, load_config)
```

位置：`model_loader/weight_utils.py:240`

它按顺序查：

```text
1. model_config.hf_config.quantization_config；
2. hf_config.text_config.quantization_config；
3. hf_config.compression_config；
4. hf_overrides["quantization_config_file"]；
5. hf_overrides["quantization_config_dict_json"]；
6. model_config.quantization_config，用于 online quantization；
7. bitsandbytes 的 inflight config；
8. 本地 / 远程 quant config 文件。
```

位置：`weight_utils.py:247` 到 `weight_utils.py:335`

如果 model_config 没有 quantization：

```text
Model quantization method is not specified in the config.
```

位置：`weight_utils.py:243` 到 `weight_utils.py:245`

如果 `hf_overrides` 不是 dict：

```text
hf_overrides must be a dict for get_quant_config to get the quantization config from it.
```

位置：`weight_utils.py:293` 到 `weight_utils.py:300`

如果指定了 config file，但 quant class 没有实现：

```text
from_config_file is specified in hf_override config, but quant_cls.from_config_file is not implemented
```

位置：`weight_utils.py:301` 到 `weight_utils.py:310`

---

## 7. checkpoint 权重缺失如何被发现

权重加载后，默认 loader 会检查哪些参数没初始化。

入口：

```python
DefaultModelLoader.track_weights_loading()
```

位置：`default_loader.py:447`

逻辑：

```text
1. 收集 model.named_parameters() 中所有应该加载的参数；
2. 对 online quant / process_weights_after_loading 创建的参数做豁免；
3. 对 KV cache scale 等后处理参数做豁免；
4. 如果还有未加载参数，则报错。
```

报错：

```text
Following weights were not initialized from checkpoint: {...}
```

位置：`default_loader.py:465` 到 `default_loader.py:470`

这类错误常见原因：

```text
1. checkpoint 的 qweight / qzeros / scales 名字和 vLLM 预期不一致；
2. fused module 的 shard_id 映射不对；
3. HF config 的 quant_method 被识别成了错误后端；
4. modules_to_not_convert / ignored_layers 和实际模型结构不匹配；
5. KV cache scale 名称没有成功 remap。
```

---

## 8. KV cache quantization 的特殊限制

KV cache quantization 和 weight quantization 是两套链路。

### 8.1 cache dtype 配置

配置位置：`config/cache.py`

关键字段：

```text
cache_dtype：
  KV cache 的 dtype，例如 auto / fp8 / nvfp4 等。

calculate_kv_scales：
  已废弃；以前用于动态计算 k_scale / v_scale。

kv_cache_dtype_skip_layers：
  跳过某些层的 KV cache quantization。
```

位置：`cache.py:110` 到 `cache.py:117`

如果启用 quantized KV cache，会有日志提醒：

```text
Using xxx data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. Meanwhile, it may cause accuracy drop without a proper scaling factor
```

位置：`cache.py:268` 到 `cache.py:286`

如果使用 per-token-head scales，会提示：

```text
Dynamic per-token-head scales will be computed at runtime.
```

位置：`cache.py:271` 到 `cache.py:276`

### 8.2 calculate_kv_scales 已废弃

如果设置 `--calculate-kv-scales`：

```text
The `--calculate-kv-scales` option is deprecated and will be removed in v0.19. The scales will be loaded from the model checkpoint if available, otherwise they default to 1.0.
```

位置：`cache.py:256` 到 `cache.py:266`

这意味着调试 FP8 KV cache 精度时要关注：

```text
1. checkpoint 是否有 k_scale / v_scale；
2. 名字是否被正确 remap；
3. 没有 scale 时是否默认 1.0；
4. 是否需要跳过某些层的 kv_cache_dtype。
```

### 8.3 KV scale 名称 remap

入口：

```python
maybe_remap_kv_scale_name(name, params_dict)
```

位置：`weight_utils.py:1341`

它支持把不同 checkpoint 格式映射到 vLLM 期望的：

```text
.attn.k_scale
.attn.v_scale
.attn.q_scale
.attn.k_zero_point
.attn.v_zero_point
```

如果 checkpoint 仍使用旧的 `.kv_scale`：

```text
DEPRECATED. Found kv_scale in the checkpoint. This format is deprecated in favor of separate k_scale and v_scale tensors...
```

位置：`weight_utils.py:1363` 到 `weight_utils.py:1370`

如果 remap 后模型里找不到对应参数：

```text
Found kv_scale in the checkpoint ..., but not found the expected name in the model ... kv_scale is not loaded.
```

位置：`weight_utils.py:1373` 到 `weight_utils.py:1379`

### 8.4 wake_up 后 FP8 KV scale 重置

`GPUModelRunner.post_kv_cache_wake_up()` 会调用：

```python
self.init_fp8_kv_scales()
```

位置：`gpu_model_runner.py:935` 到 `gpu_model_runner.py:936`

它会：

```text
1. 如果不是 quantized KV cache，直接返回；
2. 把 KV cache tensor 清零；
3. 把 attention layer 的 _k_scale / k_scale 重置为 1.0；
4. 把 _v_scale / v_scale 重置为 1.0。
```

位置：`gpu_model_runner.py:939` 到 `gpu_model_runner.py:980`

注释里说明原因：

```text
如果 wake_up 后 scale 还是 0.0，KV cache 值会被等效清零，导致 gibberish output。
```

位置：`gpu_model_runner.py:941` 到 `gpu_model_runner.py:945`

### 8.5 NVFP4 KV cache 与 MLA 不兼容

配置校验：

```python
validate_nvfp4_kv_cache_with_mla()
```

位置：`config/vllm.py:2160`

如果 `cache_dtype == "nvfp4"` 且模型使用 MLA：

```text
nvfp4 KV cache is not supported with MLA (Multi-head Latent Attention) backends. Please use a different --kv-cache-dtype (e.g., 'fp8' or 'auto') for MLA models such as DeepSeek.
```

位置：`config/vllm.py:2164` 到 `config/vllm.py:2169`

---

## 9. FP8 常见限制

FP8 配置类：

```text
Fp8Config
```

位置：`fp8.py:99`

### 9.1 activation_scheme 只支持 static / dynamic

```python
ACTIVATION_SCHEMES = ["static", "dynamic"]
```

位置：`fp8.py:94`

如果传了其他值：

```text
Unsupported activation scheme xxx
```

位置：`fp8.py:114` 到 `fp8.py:115`

### 9.2 block-wise FP8 的限制

如果配置了 `weight_block_size`：

```text
1. 只支持 fp8-serialized checkpoint；
2. weight_block_size 必须是 2 维；
3. activation_scheme 必须是 dynamic。
```

对应错误：

```text
The block-wise quantization only supports fp8-serialized checkpoint for now.
```

```text
The quantization block size of weight must have 2 dimensions...
```

```text
The block-wise quantization only supports dynamic activation scheme for now...
```

位置：`fp8.py:119` 到 `fp8.py:135`

### 9.3 ignored_layers / modules_to_not_convert

FP8 会读取：

```text
ignored_layers
modules_to_not_convert
```

位置：`fp8.py:163` 到 `fp8.py:170`

在 `get_quant_method()` 中，如果 layer 被 skip：

```text
LinearBase → UnquantizedLinearMethod
RoutedExperts → UnquantizedFusedMoEMethod
```

位置：`fp8.py:182` 到 `fp8.py:207`

因此调试“为什么某层没量化”时，要看：

```text
quant_config.ignored_layers
quant_config.packed_modules_mapping
layer prefix 是否命中 is_layer_skipped()
```

### 9.4 FP8 online / offline 差异

如果 checkpoint 不是 fp8 serialized：

```text
LinearBase → Fp8PerTensorOnlineLinearMethod
```

位置：`fp8.py:189` 到 `fp8.py:196`

如果 checkpoint 是 fp8 serialized：

```text
LinearBase → Fp8LinearMethod
```

位置：`fp8.py:197` 到 `fp8.py:200`

这会影响：

```text
1. 权重是否在加载后现场量化；
2. checkpoint 是否必须有 weight_scale；
3. process_weights_after_loading() 做什么；
4. memory peak 和加载耗时。
```

---

## 10. AWQ 常见限制和 fallback

配置类：

```text
AutoAWQConfig
```

位置：`auto_awq.py`

### 10.1 weight_bits 限制

AWQ 初始化时会检查：

```python
if self.weight_bits not in self.TYPE_MAP:
    raise ValueError(...)
```

错误：

```text
Unsupported num_bits = x. Supported: ... For 8-bit AWQ, use Marlin backend by setting backend='awq:marlin' or backend='marlin'.
```

位置：`auto_awq.py:200` 到 `auto_awq.py:207`

### 10.2 AWQ Marlin fallback

AWQ 会优先尝试 Marlin：

```text
not VLLM_BATCH_INVARIANT
current_platform.is_cuda()
check_marlin_supported(quant_type, group_size, zero_point)
```

位置：`auto_awq.py:307` 到 `auto_awq.py:315`

如果当前 layer shape 不支持 AutoAWQMarlin：

```text
Layer '%s' is not supported by AutoAWQMarlin. Falling back to unoptimized AWQ kernels.
```

位置：`auto_awq.py:319` 到 `auto_awq.py:327`

### 10.3 AWQ MoE fallback

如果 MoE layer 不支持 AutoAWQMoEMarlin：

```text
Layer 'xxx' is not supported by AutoAWQMoEMarlin. Falling back to Moe WNA16 kernels.
```

位置：`auto_awq.py:342` 到 `auto_awq.py:353`

这类不是致命错误，而是性能 fallback。

---

## 11. GPTQ 常见限制和 fallback

配置类：

```text
AutoGPTQConfig
```

位置：`auto_gptq.py`

### 11.1 bits / sym 限制

初始化时检查：

```python
if (weight_bits, is_sym) not in self.TYPE_MAP:
    raise ValueError(...)
```

错误：

```text
Unsupported quantization config: bits=x, sym=y
```

位置：`auto_gptq.py:157` 到 `auto_gptq.py:160`

### 11.2 GPTQ MoE fallback

如果 MoE layer 不支持 GPTQMoeMarlin：

```text
Layer 'xxx' is not supported by GPTQMoeMarlin. Falling back to Moe WNA16 kernels.
```

位置：`auto_gptq.py:243` 到 `auto_gptq.py:253`

### 11.3 GPTQ override 顺序

`ModelConfig._verify_quantization()` 中，GPTQ 相关 override 按顺序检查：

```text
auto_gptq
gptq
gptq_marlin
```

位置：`config/model.py:983` 到 `config/model.py:987`

原因是：

```text
同一个 checkpoint 的 quant_method 可能是 gptq，
但 vLLM 会根据平台和配置选择更合适的实现。
```

如果用户强行指定了不匹配的 quantization，会触发配置不一致错误。

---

## 12. BitsAndBytes 常见限制

配置类：

```text
BitsAndBytesConfig
```

位置：`bitsandbytes.py`

### 12.1 4bit storage 限制

当前只支持：

```text
bnb_4bit_quant_storage == "uint8"
```

否则报：

```text
Unsupported bnb_4bit_quant_storage: xxx
```

位置：`bitsandbytes.py:80` 到 `bitsandbytes.py:83`

### 12.2 8bit 不支持 CUDA graph

如果是 bitsandbytes 8bit，并且没有 enforce eager：

```text
CUDA graph is not supported on BitsAndBytes 8bit yet, fallback to the eager mode.
```

位置：`config/model.py:1090` 到 `config/model.py:1110`

这类问题表现为：

```text
没有报错，但运行模式从 CUDA graph 退回 eager；
性能可能下降；
日志里会看到 fallback。
```

### 12.3 skip module

BnB 支持：

```text
llm_int8_skip_modules
```

如果 prefix 命中 skip 规则：

```text
LinearBase → UnquantizedLinearMethod
```

位置：`bitsandbytes.py:160` 到 `bitsandbytes.py:168`

判断逻辑：`bitsandbytes.py:178` 到 `bitsandbytes.py:192`

---

## 13. Online quantization 常见限制

配置类：

```text
OnlineQuantizationConfig
```

位置：`online/base.py:74`

### 13.1 必须指定 linear 或 moe

如果没有指定任何对象：

```text
OnlineQuantizationConfig requires at least one of quantization_config.linear or quantization_config.moe to be set.
```

位置：`online/base.py:83` 到 `online/base.py:88`

### 13.2 不能从 checkpoint quant config 加载

`from_config()` 直接报：

```text
OnlineQuantizationConfig does not support loading from a checkpoint config. Use quantization_config or quantization='fp8_per_tensor'/'fp8_per_block' instead.
```

位置：`online/base.py:110` 到 `online/base.py:116`

### 13.3 weight QuantKey 支持有限

online linear 支持的 weight key：

```text
kFp8StaticTensorSym
kFp8Static128BlockSym
kFp8StaticChannelSym
kMxfp8Dynamic
```

位置：`online/base.py:58` 到 `online/base.py:63`

online MoE 额外支持：

```text
kInt8StaticChannelSym
```

位置：`online/base.py:65` 到 `online/base.py:71`

如果传了不支持的 weight key：

```text
online quantization for LinearBase/RoutedExperts with weight=... is not supported; supported weight keys: ...
```

位置：`online/base.py:126` 到 `online/base.py:132`

### 13.4 activation override 暂不支持

如果用户显式设置 activation override：

```text
activation override (activation=...) is not yet supported for online XxxMethod
```

位置：`online/base.py:134` 到 `online/base.py:140`

### 13.5 FP8 PTPC online 不能走 MarlinFP8

`Fp8PtpcOnlineLinearMethod` 要求 kernel 真正支持 per-token activation quant。

如果选到了 MarlinFP8：

```text
FP8 PTPC online quant requires a kernel that honors per-token activation quantization; MarlinFP8 is W8A16 weight-only. Requires SM89+ for Cutlass FP8 or ROCm MI3xx for rowwise scaled_mm.
```

位置：`online/fp8.py:322` 到 `online/fp8.py:330`

---

## 14. kernel 选择的两级判断

量化 kernel 抽象在：

```text
model_executor/kernels/linear/base.py
```

核心接口：

```text
is_supported(compute_capability)
can_implement(config)
```

位置：`base.py:190` 到 `base.py:231`

区别是：

```text
is_supported：
  当前硬件 / 平台能不能跑这个 kernel。

can_implement：
  当前 quantization config、shape、scale 粒度能不能被这个 kernel 实现。
```

这两个判断构成调试 kernel fallback 的核心：

```text
硬件不支持：
  换 GPU / 平台 / 安装依赖 / 开 env。

配置不支持：
  换 quantization_config / group_size / dtype / backend / batch invariant。
```

---

## 15. Marlin 常见限制

Marlin 是很多 W4A16 / W8A16 / FP8 fallback 的优化 kernel。

### 15.1 mixed precision Marlin

文件：`mixed_precision/marlin.py`

限制：

```text
1. 只支持 CUDA；
2. quant type 必须在 query_marlin_supported_quant_types() 中；
3. group_size 必须在 MARLIN_SUPPORTED_GROUP_SIZES 中；
4. act-order / g_idx 场景需要严格 shape 检查；
5. group 不能跨 TP partition；
6. W8A8 不支持 Marlin kernel。
```

对应代码：`mixed_precision/marlin.py:40` 到 `mixed_precision/marlin.py:83`

典型错误：

```text
Marlin only supported on CUDA
```

```text
Quant type (...) not supported by Marlin
```

```text
Group size (...) not supported by Marlin
```

```text
in_features per partition ... is not divisible by group_size = ...
```

W8A8 断言：

```text
W8A8 is not supported by marlin kernel.
```

位置：`mixed_precision/marlin.py:91` 到 `mixed_precision/marlin.py:96`

### 15.2 FP8 Marlin

文件：`scaled_mm/marlin.py`

FP8 Marlin 定位是：

```text
给缺少 FP8 硬件支持的 GPU 提供 fast weight-only FP8 path。
```

位置：`scaled_mm/marlin.py:29` 到 `scaled_mm/marlin.py:33`

限制：

```text
1. 只支持 CUDA；
2. 需要 compute capability >= 7.5；
3. 不支持 VLLM_BATCH_INVARIANT；
4. 高能力 GPU 默认不走 FP8 Marlin，除非设置 VLLM_TEST_FORCE_FP8_MARLIN=1。
```

位置：`scaled_mm/marlin.py:35` 到 `scaled_mm/marlin.py:56`

注意：

```text
FP8 Marlin 是 W8A16 / weight-only 风格 fallback，
不等价于真正 W8A8 per-token activation FP8。
```

---

## 16. Cutlass / DeepGEMM / FlashInfer FP8 限制

### 16.1 Cutlass FP8

普通 Cutlass FP8 scaled mm：

```text
requires CUDA
```

位置：`scaled_mm/cutlass.py:163` 到 `scaled_mm/cutlass.py:169`

它会对权重 K/N 做 16 对齐 padding：

```text
pad_k = (16 - weight.shape[0] % 16) % 16
pad_n = (16 - weight.shape[1] % 16) % 16
```

位置：`scaled_mm/cutlass.py:211` 到 `scaled_mm/cutlass.py:221`

这解释了为什么有些 shape 可以通过 padding 支持，而不是直接报错。

### 16.2 Cutlass block FP8

block FP8 Cutlass 限制：

```text
1. 设备 compute capability 必须支持 CUTLASS_BLOCK_FP8；
2. activation quant 必须是 dynamic per-token-group；
3. group_shape 必须是 (1, 128)。
```

典型原因：

```text
Supports only dynamic per token group activation quantization with group_shape=(1,128).
```

位置：`scaled_mm/cutlass.py:287` 到 `scaled_mm/cutlass.py:310`

### 16.3 DeepGEMM FP8 block

文件：`scaled_mm/deep_gemm.py`

限制：

```text
1. 只支持 CUDA；
2. 目前只支持 Hopper / Blackwell；
3. output dtype 只支持 bfloat16；
4. activation group_shape 必须是 (1, 128)；
5. 某些 model_type 会自动禁用；
6. metadata / weight_shape 必须满足 should_use_deepgemm_for_fp8_linear()。
```

位置：`deep_gemm.py:46` 到 `deep_gemm.py:82`

典型原因：

```text
Currently, only Hopper and Blackwell GPUs are supported.
```

```text
Supports only output dtype of bfloat16
```

```text
Supports only dynamic per token group activation quantization with group_shape=(1,128).
```

```text
The provided metadata is not supported.
```

### 16.4 FlashInfer FP8

普通 FlashInfer FP8 scaled mm 限制：

```text
1. requires CUDA；
2. requires FlashInfer to be installed；
3. compute capability >= 100；
4. activation 和 weight 都必须是 per-tensor scale。
```

位置：`scaled_mm/flashinfer.py:39` 到 `scaled_mm/flashinfer.py:65`

block-scale FlashInfer FP8 限制：

```text
1. 只支持 CUDA；
2. FlashInfer block-scale FP8 GEMM 必须可用；
3. activation group_shape 必须是 (1,128)；
4. metadata 必须满足 should_use_flashinfer_for_blockscale_fp8_gemm()。
```

位置：`scaled_mm/flashinfer.py:88` 到 `scaled_mm/flashinfer.py:131`

---

## 17. ROCm FP8 限制

文件：`scaled_mm/rocm.py`

ROCm FP8 scaled mm 限制：

```text
1. requires ROCm；
2. requires MI3xx or gfx12x；
3. requires VLLM_ROCM_USE_SKINNY_GEMM；
4. activation scale 必须 per-tensor；
5. weight scale 必须 per-tensor。
```

位置：`scaled_mm/rocm.py:74` 到 `scaled_mm/rocm.py:102`

典型原因：

```text
requires MI3xx or gfx12x
```

```text
requires VLLM_ROCM_USE_SKINNY_GEMM to be enabled.
```

```text
requires per tensor activation and weight scales.
```

此外 ROCm 平台里也有模型架构限制，例如某些 model architecture 不支持，会在：

```text
platforms/rocm.py
```

里通过 `verify_model_arch()` 抛出。

---

## 18. CUDA graph / torch.compile 相关限制

### 18.1 BitsAndBytes 8bit 强制 eager

前面提到：

```text
CUDA graph is not supported on BitsAndBytes 8bit yet, fallback to the eager mode.
```

位置：`config/model.py:1090` 到 `config/model.py:1110`

### 18.2 Encoder-decoder on ROCm fallback eager

`ModelConfig._verify_cuda_graph()` 中：

```text
CUDA graph is not supported for encoder-decoder models on ROCm yet, fallback to eager mode.
```

位置：`config/model.py:1073` 起

这不是量化专属，但量化调试时常和性能 fallback 混在一起。

### 18.3 quant custom op 与 compile

许多 fused quant pass 会使用 custom op 和 `auto_functionalized()`。

当 compile 失败或 fallback eager 时，要区分：

```text
1. 是 quant kernel 本身不支持；
2. 还是 torch.compile 对 mutating custom op / fake impl / graph capture 不支持；
3. 还是 CUDA graph 对当前动态形状不支持。
```

调试建议：

```text
先用 enforce_eager / 关闭 compile 复现正确性，
再逐步打开 compile / cudagraph 看是哪一层出问题。
```

---

## 19. layer 层调试：怎么确认每层用了什么 quant_method

所有 linear 层最终在：

```python
LinearBase.forward()
```

调用：

```python
output = self.quant_method.apply(self, x, bias)
```

位置：`linear.py:372` 到 `linear.py:379`

因此调试一层是否量化，要看：

```text
layer.quant_method 的类型
layer.weight / qweight / qzeros / weight_scale 是否存在
layer.input_scale 是否存在
layer.weight_block_size 是否存在
quant_method.process_weights_after_loading 是否执行过
```

常见判断：

```text
UnquantizedLinearMethod：
  该层没有量化，可能被 ignore / skip。

Fp8LinearMethod：
  FP8 checkpoint path。

Fp8PerTensorOnlineLinearMethod：
  online FP8 per-tensor weight path。

Fp8PerBlockOnlineLinearMethod：
  online FP8 block weight path。

AutoAWQMarlinLinearMethod：
  AWQ 走 Marlin。

AutoAWQLinearMethod：
  AWQ fallback 到非 Marlin kernel。

AutoGPTQLinearMethod：
  GPTQ path。

BitsAndBytesLinearMethod：
  BnB path。
```

---

## 20. 参数层调试：看哪些 tensor

### 20.1 FP8 linear

常见参数：

```text
weight：
  FP8 weight 或加载后在线量化出来的 FP8 weight。

weight_scale：
  per-tensor / per-channel weight scale。

weight_scale_inv：
  block FP8 / DeepSeek 风格 checkpoint 可能使用。

input_scale：
  static activation scale；dynamic 时通常为 None。

input_scale_ub：
  dynamic per-token scale upper bound，某些路径使用。

weight_block_size：
  block FP8 的 block shape。
```

### 20.2 GPTQ / AWQ / Marlin

常见参数：

```text
qweight / weight_packed：
  packed quantized weight。

qzeros / w_zp：
  zero point。

scales / weight_scale：
  group scale。

g_idx：
  GPTQ act-order 分组索引。

workspace：
  Marlin workspace。
```

### 20.3 INT8 / W8A8

常见参数：

```text
weight：int8 weight
weight_scale：weight scale
input_scale：static input scale，dynamic 时可为空
input_zero_point / azp_adj：asymmetric path 使用
```

### 20.4 KV cache quant

常见参数：

```text
attn.k_scale / _k_scale
attn.v_scale / _v_scale
attn.q_scale
k_zero_point / v_zero_point / q_zero_point
```

---

## 21. 常见错误分类

### 21.1 配置错误

```text
Invalid quantization method: xxx
Unknown quantization method: xxx
Quantization method specified in the model config (...) does not match ...
quantization_config is only supported when quantization is one of ...
```

定位：

```text
quantization/__init__.py
config/model.py
config/quantization.py
```

### 21.2 平台 / capability 错误

```text
xxx quantization is currently not supported in yyy.
The quantization method xxx is not supported for the current GPU.
Minimum capability: y. Current capability: z.
```

定位：

```text
platforms/interface.py
config/vllm.py
各 QuantizationConfig.get_min_capability()
```

### 21.3 dtype 错误

```text
torch.xxx is not supported for quantization method xxx. Supported dtypes: [...]
```

定位：

```text
config/vllm.py
QuantizationConfig.get_supported_act_dtypes()
```

### 21.4 checkpoint 缺字段

```text
Cannot find any of [...] in the model's quantization config.
Following weights were not initialized from checkpoint: {...}
```

定位：

```text
base_config.py:get_from_keys()
model_loader/default_loader.py:track_weights_loading()
```

### 21.5 kernel 不支持 / fallback

```text
Layer 'xxx' is not supported by AutoAWQMarlin. Falling back...
Layer 'xxx' is not supported by GPTQMoeMarlin. Falling back...
requires FlashInfer to be installed.
requires compute capability 100 and above.
Supports only dynamic per token group activation quantization with group_shape=(1,128).
```

定位：

```text
quantization/auto_awq.py
quantization/auto_gptq.py
kernels/linear/*
```

### 21.6 runtime / 精度问题

表现：

```text
输出乱码 / gibberish output
NaN logits
吞吐下降
CUDA graph fallback eager
量化后困惑度明显变差
```

优先看：

```text
KV cache scale 是否加载 / 重置；
activation scale 是否 static/dynamic 正确；
是否有 fallback 到非预期 backend；
是否用了不合适的 cache_dtype；
是否有 layer 被 skip；
是否存在 dtype 自动转换。
```

---

## 22. 推荐调试路线

### 22.1 第一步：确认最终量化方法

看：

```text
model_config.quantization
model_config.quantization_config
hf_config.quantization_config
hf_config.compression_config
```

关键源码：

```text
config/model.py:970
model_loader/weight_utils.py:240
```

要确认：

```text
1. CLI 传入的 --quantization 是什么；
2. checkpoint config 里的 quant_method 是什么；
3. 是否发生 override；
4. 最终 self.quantization 是什么。
```

### 22.2 第二步：确认平台和 dtype

看：

```text
current_platform.device_name
current_platform.supported_quantization
current_platform.get_device_capability()
model_config.dtype
quant_config.get_min_capability()
quant_config.get_supported_act_dtypes()
```

关键源码：

```text
platforms/interface.py:824
config/vllm.py:609
```

### 22.3 第三步：确认 quant config 是否正确解析

看：

```text
quant_config.__repr__()
quant_config.get_name()
quant_config.ignored_layers / modules_to_not_convert
weight_bits / group_size / zero_point / desc_act / activation_scheme
weight_block_size / store_dtype
```

关键源码：

```text
fp8.py
access_awq.py / auto_awq.py
auto_gptq.py
bitsandbytes.py
online/base.py
```

### 22.4 第四步：确认每层 quant_method

遍历模型模块：

```text
for name, module in model.named_modules():
    if hasattr(module, "quant_method"):
        print(name, type(module.quant_method), module.quant_method)
```

重点检查：

```text
1. 预期量化的层是否真的量化；
2. 是否被 UnquantizedLinearMethod 替代；
3. 是否从 Marlin fallback 到普通 kernel；
4. MoE 是否 fallback 到 MoeWNA16；
5. lm_head 是否量化。
```

### 22.5 第五步：确认参数是否加载

重点检查：

```text
weight / qweight
weight_scale / scales
qzeros / zero_point
g_idx
input_scale
weight_scale_inv
k_scale / v_scale
```

如果报未初始化：

```text
Following weights were not initialized from checkpoint
```

要沿着：

```text
checkpoint tensor name
  → weight mapper
  → packed_modules_mapping
  → parameter weight_loader
  → loaded_weights set
```

逐层查。

### 22.6 第六步：确认 kernel 选择和 fallback

重点看：

```text
kernel.is_supported()
kernel.can_implement()
warning_once 日志
process_weights_after_loading()
```

如果是 FP8：

```text
CutlassFP8ScaledMMLinearKernel
MarlinFP8ScaledMMLinearKernel
DeepGemmFp8BlockScaledMMKernel
FlashInferFP8ScaledMMLinearKernel
ROCmFP8ScaledMMLinearKernel
```

如果是 AWQ/GPTQ：

```text
AutoAWQMarlinLinearMethod vs AutoAWQLinearMethod
AutoGPTQLinearMethod
MarlinLinearKernel.can_implement()
```

### 22.7 第七步：缩小复现

建议按顺序缩小：

```text
1. 单卡 TP=1；
2. enforce_eager；
3. 关闭 CUDA graph / compile；
4. 小 batch、短 prompt；
5. 禁用 KV cache quantization；
6. 对比同模型 fp16/bf16 输出；
7. 逐层检查量化参数和 scale。
```

---

## 23. 精度问题怎么定位

量化精度问题常见不是 crash，而是：

```text
生成质量变差；
logits 分布异常；
出现 NaN；
长上下文退化；
开启 fp8 kv cache 后明显变差。
```

建议顺序：

```text
1. 先确认 weight quant 是否正确加载；
2. 再确认 activation scale 是 static 还是 dynamic；
3. 再确认 KV cache dtype 和 k/v scale；
4. 再确认是否有 layer 被意外 skip；
5. 再确认是否 fallback 到了不同 kernel；
6. 最后对比 fp16/bf16 baseline。
```

对 FP8 KV cache 特别注意：

```text
没有合适 scaling factor 时会有 accuracy drop；
checkpoint 没有 k_scale/v_scale 时可能默认 1.0；
wake_up 后 scale 会被重置为 1.0；
nvfp4 KV cache 不支持 MLA。
```

---

## 24. 性能问题怎么定位

量化后性能不升反降，常见原因：

```text
1. 目标 kernel 没选上，fallback 到慢 kernel；
2. BitsAndBytes 8bit fallback eager；
3. Marlin 不支持当前 group_size / shape；
4. FlashInfer / DeepGEMM 依赖没安装或 metadata 不支持；
5. FP8 dynamic activation quant 的额外 kernel 开销超过 GEMM 收益；
6. batch 太小，量化和 scale 计算开销占比高；
7. TP shard 后 in_features 不再能被 group_size 整除；
8. CUDA graph / compile 没启用或失败。
```

建议看：

```text
1. 日志里的 warning_once；
2. 每层 quant_method 类型；
3. kernel class 类型；
4. cudagraph_stats；
5. profiler / NVTX；
6. 是否 enforce_eager 被自动打开。
```

---

## 25. 容易疑惑的点

### 25.1 `--quantization fp8` 一定代表 checkpoint 是 FP8 吗？

不一定。

在 FP8 config 中：

```text
is_checkpoint_fp8_serialized = "fp8" in quant_method
```

如果 checkpoint 不是 FP8 serialized，vLLM 可能走 online FP8，把 fp16/bf16 weight 在加载后量化。

### 25.2 fallback 是错误吗？

不一定。

例如：

```text
AWQ Marlin 不支持某层 → fallback 到 unoptimized AWQ；
GPTQ MoE Marlin 不支持 → fallback 到 MoeWNA16；
BitsAndBytes 8bit 不支持 CUDA graph → fallback eager。
```

这些通常是性能变化，不一定是正确性错误。

### 25.3 为什么同一个模型在不同 GPU 上量化后行为不同？

因为量化后端依赖：

```text
compute capability；
FP8 tensor core 支持；
Marlin 支持；
FlashInfer / DeepGEMM 是否可用；
ROCm / CUDA 平台差异；
shape padding / group_size 限制。
```

同一个 `quantization_config` 可能在不同硬件上选到不同 kernel。

### 25.4 为什么 checkpoint 里有 scale 但 vLLM 说没加载？

常见原因：

```text
1. scale 名称不是 vLLM 预期格式；
2. mapper 没有覆盖该模型结构；
3. fused QKV / gate_up_proj 的 shard_id 不匹配；
4. KV scale 需要通过 maybe_remap_kv_scale_name() 映射；
5. layer 被 ignore，相关参数不会创建。
```

### 25.5 为什么 FP8 KV cache 输出乱码？

优先看：

```text
1. k_scale / v_scale 是否正确加载；
2. 是否默认 1.0；
3. wake_up 后是否重置；
4. cache_dtype 是否和 attention backend 兼容；
5. 是否使用 nvfp4 + MLA 这种不支持组合。
```

### 25.6 `QuantKey` 有什么调试价值？

`QuantKey` 描述：

```text
量化 dtype；
scale 是 static 还是 dynamic；
scale 粒度是 per-tensor / per-token / per-channel / per-group；
是否有二级 scale；
是否 symmetric。
```

很多 kernel 的 `can_implement()` 就是在检查 `QuantKey`。

---

## 26. 最小调试清单

如果只想快速定位，可以按这个清单：

```text
1. 打印 model_config.quantization。
2. 打印 hf_config.quantization_config / compression_config。
3. 打印 VllmConfig.quant_config。
4. 确认 current_platform.supported_quantization。
5. 确认 GPU capability >= quant_config.get_min_capability()。
6. 确认 model dtype 在 get_supported_act_dtypes() 中。
7. 遍历 layer.quant_method 类型。
8. 检查关键参数：qweight / scales / qzeros / input_scale / k_scale / v_scale。
9. 搜索日志 warning_once：fallback / not supported / deprecated。
10. 用 enforce_eager + 小 batch 复现正确性。
11. 关闭 kv_cache_dtype quantization 对比精度。
12. 和 fp16/bf16 baseline 对比 logits 或短输出。
```

---

## 27. 总结

vLLM 量化限制可以压缩成：

```text
配置要匹配 checkpoint；
checkpoint 要匹配 QuantizationConfig；
QuantizationConfig 要能给每层创建 quant_method；
quant_method 要能加载对应参数；
kernel 要同时满足硬件和 shape/scale 粒度约束；
runtime 还要和 CUDA graph、torch.compile、KV cache dtype 兼容。
```

调试主线是：

```text
ModelConfig._verify_quantization()
  → VllmConfig._get_quantization_config()
  → get_quant_config()
  → QuantizationConfig.from_config()
  → QuantizationConfig.get_quant_method()
  → layer.quant_method.create_weights()
  → weight_loader / track_weights_loading()
  → process_weights_after_loading()
  → kernel.is_supported() / kernel.can_implement()
  → quant_method.apply()
```

一句话压缩：

```text
量化调试不要直接从 kernel crash 开始猜；先沿着“配置 → checkpoint → layer 参数 → kernel 选择 → runtime 输出”逐层确认，绝大多数 unsupported、fallback、精度和性能问题都能在这条链路上定位。
```
