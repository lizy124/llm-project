# 01. Quantization 在 vLLM 中负责什么？

源码位置：

- `vllm/vllm/config/quantization.py`
- `vllm/vllm/config/model.py`
- `vllm/vllm/config/vllm.py`
- `vllm/vllm/config/cache.py`
- `vllm/vllm/model_executor/model_loader/weight_utils.py`
- `vllm/vllm/model_executor/model_loader/utils.py`
- `vllm/vllm/model_executor/layers/quantization/`
- `vllm/vllm/model_executor/layers/linear.py`
- `vllm/vllm/model_executor/layers/fused_moe/`
- `vllm/vllm/model_executor/layers/attention/attention.py`
- `vllm/vllm/_custom_ops.py`

本问题关注：量化在 vLLM 中的边界，区分权重量化、激活量化、KV cache 量化和 kernel 支持；它如何从 CLI / HF config 进入 `ModelConfig` / `VllmConfig`，如何影响 Linear / MoE / Attention 层创建参数、加载权重、后处理权重，并最终决定运行时调用哪个低 bit kernel。

---

## 1. 一句话回答

量化在 vLLM 中不是单一开关，而是一套横跨 **配置解析、权重加载、layer 参数创建、权重后处理、kernel 选择和 KV cache dtype / scale 管理** 的执行协议。

可以先压缩成：

```text
权重量化：
  影响 Linear / MoE / Embedding 等层如何创建、加载、保存和使用权重。

激活量化：
  影响 forward 时输入 activation 是否动态 / 静态量化，以及 scale 如何传入 kernel。

KV cache 量化：
  影响 Attention 写入 / 读取 KV cache 时的 dtype、scale 和 backend 限制。

kernel 量化：
  影响运行时调用普通 GEMM，还是 Marlin / CUTLASS / Triton / FP8 / INT8 / FP4 / MoE 专用 kernel。
```

所以量化的职责不是“把模型变小”这么简单，而是：

```text
把 checkpoint 里的量化格式，转换成 vLLM layer 和 kernel 能执行的参数布局。
```

---

## 2. 最小主链路

量化从配置到执行的大致链路是：

```text
EngineArgs / HF config
  → ModelConfig.quantization / quantization_config
  → ModelConfig._verify_quantization()
  → VllmConfig._get_quantization_config()
  → weight_utils.get_quant_config()
  → QuantizationConfig.from_config()
  → model 构造时把 quant_config 传给 Linear / MoE / Attention
  → layer 调用 quant_config.get_quant_method(layer, prefix)
  → quant_method.create_weights()
  → loader 加载 checkpoint 权重 / scale / zero point
  → process_weights_after_loading()
  → forward 调用 quant_method.apply()
  → quantized kernel 执行
```

如果是 KV cache 量化，还会并行走一条 cache 配置链路：

```text
CacheConfig.cache_dtype / calculate_kv_scales / kv_cache_dtype_skip_layers
  → Attention 初始化 kv_cache_dtype 和 scale buffer
  → BaseKVCacheMethod.create_weights() 注册 q/k/v/prob scale
  → checkpoint scale 加载或默认 1.0 / runtime 计算
  → attention backend 写入 / 读取量化 KV cache
```

这两条链路经常同时存在，但边界不同：

```text
权重 / 激活量化：主要绑定 Linear / MoE / Embedding 的计算 kernel。
KV cache 量化：主要绑定 Attention 的 cache 存储格式和 scale。
```

---

## 3. 量化不负责什么

量化不直接负责以下事情：

```text
1. 不直接决定请求调度策略；
2. 不直接决定 prefill / decode 的 token budget；
3. 不直接改变 sampling 算法；
4. 不直接改变 tokenizer / detokenizer；
5. 不直接决定模型结构本身；
6. 不负责下载模型权重；
7. 不负责 KV block 的分配和回收；
8. 不负责分布式通信拓扑本身。
```

但它会间接影响这些模块能不能工作。例如：

```text
- 某种量化 kernel 不支持当前 GPU capability，会在 config 校验阶段报错；
- 某种量化方法不支持某种 activation dtype，会拒绝启动；
- fp8 KV cache 会限制 attention backend 或要求 scale；
- MoE 量化会影响 expert 权重布局和 MoE kernel 选择；
- 某些量化方法会禁用 CUDA graph 或要求 eager fallback。
```

因此量化不是调度层职责，但会给执行层和 backend 选择施加约束。

---

## 4. 配置入口：量化从哪里进入 vLLM

### 4.1 用户显式传入

用户侧常见入口是：

```text
--quantization
--quantization-config
--kv-cache-dtype
--calculate-kv-scales
--kv-cache-dtype-skip-layers
```

对应 `EngineArgs` 字段在：

- `vllm/vllm/engine/arg_utils.py:540`
- `vllm/vllm/engine/arg_utils.py:541`
- `vllm/vllm/engine/arg_utils.py:680`
- `vllm/vllm/engine/arg_utils.py:681`

这里要区分两个配置：

```text
quantization：
  选择量化方法名称，例如 awq / gptq / fp8 / compressed-tensors / online。

quantization_config：
  更细粒度描述在线量化或 activation override，例如 linear / moe / ignore。
```

### 4.2 HF config 自动识别

如果 checkpoint 的 `config.json` 或相关 config 中带有：

```text
quantization_config
compression_config
```

vLLM 会读取它并判断真实量化方法。

关键逻辑在：`vllm/vllm/config/model.py:970`

`ModelConfig._verify_quantization()` 会：

```text
1. 读取 HF quantization_config；
2. 拿到 quant_method；
3. 让各量化方法尝试 override_quantization_method()；
4. 校验用户传入的 quantization 是否和 checkpoint 匹配；
5. 校验该 quantization 是否在 QUANTIZATION_METHODS 中；
6. 调用 current_platform.verify_quantization() 校验平台支持；
7. 对 deprecated quantization 做拦截或 warning。
```

对应源码：`vllm/vllm/config/model.py:970` 到 `vllm/vllm/config/model.py:1071`

### 4.3 在线量化配置

在线量化相关配置定义在：`vllm/vllm/config/quantization.py:63`

核心对象：

```text
QuantSpec：
  描述某一类 layer 的 weight / activation 量化 spec。

QuantizationConfigArgs：
  用户可见配置，包含 linear、moe、ignore。
```

`resolve_quantization_config()` 会把：

```text
--quantization fp8_per_tensor
--quantization fp8_per_block
--quantization fp8_per_channel
--quantization mxfp8
--quantization online + --quantization-config ...
```

解析成 `QuantizationConfigArgs`。

位置：`vllm/vllm/config/quantization.py:147`

它的规则是：

```text
1. 如果 quantization 是传统 checkpoint 方法，不允许再传 quantization_config；
2. 如果 quantization 是在线量化 shorthand，会先生成 base config；
3. 如果同时传 quantization_config，则显式字段覆盖 shorthand；
4. 最终返回 QuantizationConfigArgs 或 None。
```

---

## 5. VllmConfig 里如何拿到 QuantizationConfig

`VllmConfig._get_quantization_config()` 是把 `ModelConfig` 转成 runtime `QuantizationConfig` 的入口。

位置：`vllm/vllm/config/vllm.py:609`

它做的事情是：

```text
1. 如果 model_config.quantization 不为空，调用 get_quant_config()；
2. 校验当前 GPU capability 是否满足 quant_config.get_min_capability()；
3. 校验 model_config.dtype 是否在 quant_config.get_supported_act_dtypes() 中；
4. 调用 quant_config.maybe_update_config() 做模型相关补充；
5. 返回 QuantizationConfig 实例。
```

关键点是：

```text
ModelConfig.quantization 是字符串 / 用户选择；
VllmConfig.quant_config 是真正会传给模型 layer 的 QuantizationConfig 对象。
```

也就是说，模型构造阶段一般不会直接消费字符串 `"awq"` / `"gptq"`，而是消费已经实例化的 `quant_config`。

---

## 6. QuantizationConfig / QuantizeMethodBase 抽象

量化抽象定义在：`vllm/vllm/model_executor/layers/quantization/base_config.py`

### 6.1 QuantizationConfig 负责什么

`QuantizationConfig` 是“某一种量化方法的配置对象”。

核心接口：

```text
get_name()
get_supported_act_dtypes()
get_min_capability()
get_config_filenames()
from_config(config)
override_quantization_method(...)
get_quant_method(layer, prefix)
get_cache_scale_mapper()
apply_vllm_mapper(...)
maybe_update_config(...)
is_mxfp4_quant(...)
```

位置：`vllm/vllm/model_executor/layers/quantization/base_config.py:77`

它回答的是：

```text
这个 checkpoint / 配置属于哪种量化方法？
这种量化方法支持哪些 dtype / GPU / layer？
某个具体 layer 应该用哪个 QuantizeMethod？
checkpoint 里的 scale / packed weight 名称如何映射到 vLLM layer？
```

### 6.2 QuantizeMethodBase 负责什么

`QuantizeMethodBase` 是“某个 layer 上真正执行量化逻辑的方法对象”。

核心接口：

```text
create_weights(layer, ...)
apply(layer, ...)
process_weights_after_loading(layer)
embedding(layer, ...)
tie_weights(layer, ...)
```

位置：`vllm/vllm/model_executor/layers/quantization/base_config.py:19`

它回答的是：

```text
这个 layer 应该注册哪些参数？
权重加载后是否需要 repack / transpose / quantize？
forward 时调用哪个 kernel？
```

### 6.3 两者的边界

可以这样理解：

```text
QuantizationConfig：
  全局配置 / checkpoint 格式 / 方法选择。

QuantizeMethodBase：
  单个 layer 的参数创建 / 权重后处理 / forward 执行。
```

也就是说：

```text
quant_config.get_quant_method(layer, prefix)
```

是全局配置进入具体 layer 的分界线。

---

## 7. 支持的量化方法注册在哪里

量化方法列表在：`vllm/vllm/model_executor/layers/quantization/__init__.py:12`

`QuantizationMethods` 包含多类方法，例如：

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

`get_quantization_config()` 会把量化名称映射到具体 `QuantizationConfig` 子类。

位置：`vllm/vllm/model_executor/layers/quantization/__init__.py:108`

例如：

```text
"awq"                  → AutoAWQConfig
"gptq"                 → AutoGPTQConfig
"compressed-tensors"   → CompressedTensorsConfig
"bitsandbytes"         → BitsAndBytesConfig
"fp8"                  → Fp8Config
"online"               → OnlineQuantizationConfig
```

它还有一个扩展点：

```python
register_quantization_config("my_quant")
```

位置：`vllm/vllm/model_executor/layers/quantization/__init__.py:58`

这说明 vLLM 的量化方法不是完全写死的，可以注册自定义量化配置。

---

## 8. checkpoint 量化配置如何读取

真正读取 checkpoint 量化配置的函数是：

```python
get_quant_config(model_config, load_config)
```

位置：`vllm/vllm/model_executor/model_loader/weight_utils.py:240`

它的优先级大致是：

```text
1. 从 hf_config.quantization_config 读取；
2. 如果是 vision/text 组合模型，尝试 text_config.quantization_config；
3. compressed-tensors 可读取 compression_config；
4. compressed-tensors 会补 total_num_heads / total_num_kv_heads，方便 TP-aware scale 加载；
5. 如果 HF config 足够完整，直接 quant_cls.from_config(hf_quant_config)；
6. 否则尝试 hf_overrides 中的 quantization_config_file / dict_json；
7. 如果是在线量化，使用 model_config.quantization_config 创建 OnlineQuantizationConfig；
8. bitsandbytes inflight quantization 可用空配置；
9. 最后尝试在模型目录中查找量化方法自己的 config 文件。
```

这一层的重点是：

```text
checkpoint 描述的是“它原来如何被量化”；
vLLM 需要把它翻译成“当前 runtime 如何加载和执行”。
```

---

## 9. Linear 层如何接入量化

Linear 是权重量化最核心的接入点。

入口：`vllm/vllm/model_executor/layers/linear.py:228`

### 9.1 LinearBase 选择 quant_method

`LinearBase.__init__()` 中：

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
else:
    raise ValueError("All linear layers should support quant method.")
```

位置：`vllm/vllm/model_executor/layers/linear.py:269`

含义是：

```text
没有 quant_config：走普通 Linear；
有 quant_config：必须能为 Linear 返回一个 quant_method；
quant_method 决定该 Linear 的参数和 forward 行为。
```

### 9.2 create_weights 决定参数布局

具体 Linear 子类创建参数时，会调用：

```python
self.quant_method.create_weights(...)
```

例如 `ReplicatedLinear`：`vllm/vllm/model_executor/layers/linear.py:338`

这里不是简单注册一个 `weight`，而是由量化方法决定注册什么参数。例如不同方法可能需要：

```text
qweight
qzeros
scales
g_idx
weight_scale
input_scale
packed_weight
block_scale
marlin / cutlass 专用 packed layout
```

这也是为什么量化方法必须参与 layer 构造，而不是等 forward 时才介入。

### 9.3 forward 只调用 quant_method.apply

Linear forward 中真正执行计算的是：

```python
output = self.quant_method.apply(self, x, bias)
```

位置：`vllm/vllm/model_executor/layers/linear.py:378`

所以运行时边界非常清楚：

```text
Linear 层不关心自己是 AWQ / GPTQ / FP8 / INT8；
它只把输入交给 quant_method.apply()。
```

具体 kernel 选择被隐藏在各量化方法内部。

---

## 10. Unquantized 也是一种 Method

无量化路径并不是完全绕开量化抽象，而是使用：

```text
UnquantizedLinearMethod
```

位置：`vllm/vllm/model_executor/layers/linear.py:179`

它也实现：

```text
create_weights()
apply()
process_weights_after_loading()
```

这说明 vLLM 把普通 Linear 和量化 Linear 都统一成同一种接口：

```text
layer.quant_method.create_weights()
layer.quant_method.apply()
```

这样模型代码不需要为每种量化方法写分支。

---

## 11. 权重加载后为什么还要 process_weights_after_loading

权重从 checkpoint 加载进参数后，很多量化方法还不能直接执行，需要后处理。

统一入口在：

```python
process_weights_after_loading(model, model_config, target_device)
```

位置：`vllm/vllm/model_executor/model_loader/utils.py:100`

它会遍历模型中的模块：

```text
1. 如果模块有 quant_method，则调用 quant_method.process_weights_after_loading(module)；
2. 再单独处理 Attention / MLAAttention / MMEncoderAttention 的 post-load 逻辑；
3. 如果启用 CPU offload，会临时把参数移动到目标设备处理，再移回。
```

对应源码：`vllm/vllm/model_executor/model_loader/utils.py:103` 到 `vllm/vllm/model_executor/model_loader/utils.py:126`

后处理可能做：

```text
- weight repack；
- scale 形状整理；
- transpose；
- checkpoint layout → kernel layout；
- fp16/bf16 weight 在线量化成低 bit；
- 删除临时参数；
- 初始化 attention q/k/v/prob scale。
```

因此，量化权重加载不是：

```text
checkpoint tensor → layer.weight
```

而更像：

```text
checkpoint tensor
  → vLLM Parameter
  → quant_method.process_weights_after_loading()
  → kernel-ready layout
```

---

## 12. packed_modules_mapping 的作用

很多模型会把多个 HF 权重融合成 vLLM 的一个 packed layer，例如：

```text
q_proj + k_proj + v_proj → qkv_proj

gate_proj + up_proj → gate_up_proj
```

量化配置需要知道这种映射，否则 checkpoint 中的量化 scale / zero point 无法正确对齐到 vLLM 的 packed 参数。

相关逻辑在：

- `vllm/vllm/model_executor/model_loader/utils.py:274`
- `vllm/vllm/model_executor/utils.py:101`

`configure_quant_config()` 会把模型类上的：

```text
hf_to_vllm_mapper
packed_modules_mapping
```

传给 `quant_config`。

位置：`vllm/vllm/model_executor/model_loader/utils.py:287` 到 `vllm/vllm/model_executor/model_loader/utils.py:295`

这说明量化不仅关心 dtype，还关心：

```text
HF checkpoint 参数名如何映射到 vLLM 内部 fused / packed 模块名。
```

---

## 13. MoE 如何接入量化

MoE 的量化路径和 Linear 类似，但多了 expert、routing 和 fused kernel 维度。

入口：`vllm/vllm/model_executor/layers/fused_moe/`

### 13.1 FusedMoE 工厂接收 quant_config

`FusedMoE()` 工厂函数接收：

```python
quant_config: QuantizationConfig | None = None
```

位置：`vllm/vllm/model_executor/layers/fused_moe/layer.py:103`

它会创建 MoE 执行 pipeline：

```text
Router
  → RoutedExperts
  → MoERunner
```

其中 expert 权重在 `RoutedExperts` 中和 quant_method 绑定。

### 13.2 RoutedExperts 选择 quant_method

`RoutedExperts` 会调用：

```python
quant_config.get_quant_method(self, prefix)
```

如果没有量化方法，则使用：

```text
UnquantizedFusedMoEMethod
```

对应位置：`vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:180`

然后调用：

```python
self.quant_method.create_weights(layer=self, **moe_quant_params)
```

位置：`vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:168`

### 13.3 MoE 量化最终影响 kernel 参数

MoE 会把量化信息组织成 `FusedMoEQuantConfig`，里面可能包含：

```text
quant_dtype
weight_quant_dtype
block_shape
per_act_token_quant
per_out_ch_quant
w1_scale / w2_scale
a1_scale / a2_scale
w1_zp / w2_zp
w1_bias / w2_bias
```

这些字段最后进入 fused MoE kernel。

例如 `fused_moe.py` 中执行 kernel 时会消费 `quant_config`：

```text
use_fp8_w8a8
use_int8_w8a8
use_int8_w8a16
use_int4_w4a16
block_shape
w1_scale / w2_scale
a1_scale / a2_scale
```

位置：`vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1484`

因此 MoE 量化不只是“expert 权重低 bit”，还包含：

```text
token routing 后，如何用低 bit expert weight 做 batched expert GEMM。
```

---

## 14. Attention / KV cache 量化如何接入

KV cache 量化不是普通 Linear 权重量化，它发生在 Attention 层。

入口：`vllm/vllm/model_executor/layers/attention/attention.py`

### 14.1 初始化默认 scale

`_init_kv_cache_quant()` 会先调用：

```python
set_default_quant_scales(layer, register_buffer=True)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:153`

它初始化：

```text
_q_scale
_k_scale
_v_scale
_prob_scale
_q_scale_float
_k_scale_float
_v_scale_float
_prob_scale_float
```

这些 scale 会被 attention backend 或 KV cache 写入 kernel 使用。

### 14.2 从 quant_config 获取 KV cache quant_method

Attention 初始化时也会调用：

```python
quant_config.get_quant_method(layer, prefix=prefix)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:159`

如果该量化方法需要加载量化权重 / scale，则要求它是：

```text
BaseKVCacheMethod
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:164`

然后调用：

```python
layer.quant_method.create_weights(layer)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:188`

这一步会注册 checkpoint 可加载的 scale 参数。

### 14.3 BaseKVCacheMethod 做什么

`BaseKVCacheMethod` 定义在：`vllm/vllm/model_executor/layers/quantization/kv_cache.py:42`

它的职责是给 Attention layer 增加：

```text
q_scale
k_scale
v_scale
prob_scale
```

位置：`vllm/vllm/model_executor/layers/quantization/kv_cache.py:57`

这些参数用于：

```text
1. 写入 KV cache 前量化 key / value；
2. 从 KV cache 读取后反量化 key / value；
3. 某些 FP8 attention backend 中处理 query / probability scale。
```

### 14.4 KV scale 的来源

KV scale 可能来自三种路径：

```text
1. checkpoint 中已有 k_scale / v_scale / q_scale / prob_scale；
2. checkpoint 没有 scale，则默认使用 1.0；
3. calculate_kv_scales=True 时，runtime 根据 query/key/value 动态计算一次。
```

`BaseKVCacheMethod.process_weights_after_loading()` 会处理加载后的 scale。

位置：`vllm/vllm/model_executor/layers/quantization/kv_cache.py:74`

它会：

```text
- 如果是 per-token-head KV cache scale，则忽略 checkpoint scale，runtime 动态计算；
- 如果 KV cache 没量化，则强制 k/v scale 为 1.0；
- 如果 checkpoint 有 k_scale / v_scale，则使用它们；
- 如果只有单个 kv_scale，则复制成 k_scale 和 v_scale；
- 如果没有 scale，则默认 1.0 并给出 warning；
- 最终写入 _q_scale / _k_scale / _v_scale / _prob_scale；
- 删除临时 q_scale / k_scale / v_scale / prob_scale 参数。
```

### 14.5 runtime 计算 KV scale

Attention 里还有：

```python
def calc_kv_scales(self, query, key, value):
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:546`

它会根据当前 query / key / value 的绝对值最大值计算：

```text
_q_scale
_k_scale
_v_scale
```

然后把：

```text
calculate_kv_scales = False
```

表示只计算一次。

---

## 15. CacheConfig 与 KV cache 量化

KV cache dtype 配置定义在：`vllm/vllm/config/cache.py`

关键字段：

```text
cache_dtype
calculate_kv_scales
kv_cache_dtype_skip_layers
```

位置：

- `vllm/vllm/config/cache.py:110`
- `vllm/vllm/config/cache.py:115`

### 15.1 cache_dtype

`cache_dtype` 决定 KV cache 的存储 dtype，例如：

```text
auto
fp8
fp8_e4m3
fp8_e5m2
per-token-head scale 相关 dtype
turboquant_* dtype
```

如果是量化 KV cache，校验器会提示：

```text
它可以降低 GPU memory footprint 并提升性能，
但如果 scaling factor 不合适，可能有精度损失。
```

位置：`vllm/vllm/config/cache.py:268`

### 15.2 calculate_kv_scales

`calculate_kv_scales` 已标记 deprecated。

位置：`vllm/vllm/config/cache.py:256`

含义是：

```text
当 kv_cache_dtype 是 fp8 时，是否 runtime 动态计算 k_scale / v_scale。
如果为 False，则优先从 checkpoint 加载；没有则默认 1.0。
```

### 15.3 kv_cache_dtype_skip_layers

`kv_cache_dtype_skip_layers` 用于跳过部分层的 KV cache 量化。

位置：`vllm/vllm/config/cache.py:115`

它可以按：

```text
layer index
attention type name
```

跳过某些 attention 层，避免所有层都使用同一种 KV cache 量化策略。

---

## 16. kernel 量化到底在哪里发生

vLLM 的 Python 量化代码通常不直接完成所有低 bit 计算，而是准备参数并调用 kernel。

典型 kernel 来源包括：

```text
vllm._custom_ops
Triton kernel
CUTLASS / Marlin kernel
FlashInfer / FlashAttention backend
ROCm AITER kernel
TorchAO / bitsandbytes / modelopt 相关 kernel
MoE fused kernel
```

例如 `_custom_ops.py` 中有大量带 `kv_cache_dtype`、`k_scale`、`v_scale`、block scale、fp8/fp4 参数的接口。

相关位置：

- `vllm/vllm/_custom_ops.py:126`
- `vllm/vllm/_custom_ops.py:2515`
- `vllm/vllm/_custom_ops.py:3587`

因此量化方法通常分成两层：

```text
Python 层：
  解析配置、注册参数、加载权重、repack、准备 scale。

Kernel 层：
  真正执行 int4 / int8 / fp8 / fp4 / mxfp4 / mxfp8 计算。
```

---

## 17. 量化和并行的关系

量化权重必须和 TP / EP / PP 等并行方式配合。

### 17.1 Tensor Parallel

Linear 参数可能按输入维或输出维切分。

量化参数也必须随之切分，例如：

```text
weight shard
scale shard
zero point shard
block scale shard
packed tile shard
```

这就是为什么 `linear.py` 中有很多 shard 调整函数：

```text
adjust_marlin_shard()
adjust_block_scale_shard()
adjust_bitsandbytes_4bit_shard()
adjust_scalar_to_fused_array()
```

位置：`vllm/vllm/model_executor/layers/linear.py:70`

### 17.2 Expert Parallel / MoE

MoE 量化还要考虑：

```text
expert 在哪个 rank 上；
expert weight 是否 redundant；
是否启用 EPLB；
token routing 后如何对本地 expert 做量化 GEMM；
不同 expert 的 scale 如何加载和重排。
```

这部分主要在：

```text
vllm/vllm/model_executor/layers/fused_moe/routed_experts.py
vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py
vllm/vllm/model_executor/layers/fused_moe/fused_moe.py
```

### 17.3 Pipeline Parallel

PP 本身不改变单层量化算法，但会影响：

```text
哪些 rank 拥有哪些 layer；
哪些 layer 需要加载量化权重；
非最后 PP stage 返回 IntermediateTensors，而不是最终 logits。
```

所以量化方法主要绑定到本 rank 拥有的 layer 上。

---

## 18. 权重量化、激活量化、KV cache 量化的区别

### 18.1 权重量化

权重量化关注：

```text
checkpoint 中的 weight 如何以 int4 / int8 / fp8 / fp4 等格式存储；
加载后是否需要 repack；
forward 时如何把低 bit weight 送进 GEMM / MoE kernel。
```

主要入口：

```text
LinearBase
RoutedExperts / FusedMoE
VocabParallelEmbedding / ParallelLMHead
```

### 18.2 激活量化

激活量化关注：

```text
forward 输入 x 是否需要动态量化；
activation scale 是 per-tensor、per-token、per-channel 还是 per-block；
activation scale 是否来自 checkpoint 或 runtime 计算；
kernel 是否支持对应 activation dtype。
```

它通常和权重量化组合出现，例如：

```text
W8A8
W4A8
FP8 dynamic activation
MXFP8 activation
per-token activation scale
```

### 18.3 KV cache 量化

KV cache 量化关注：

```text
历史 key/value 存在 cache 中时用什么 dtype；
写 cache 时如何 quantize；
读 cache 时如何 dequantize；
scale 是 checkpoint 加载、默认 1.0，还是 runtime 计算；
attention backend 是否支持该 cache dtype。
```

它和 Linear 权重量化是两条不同链路。

---

## 19. 量化与模型代码的关系

模型文件通常不会直接写：

```text
if quantization == "awq": ...
```

而是把 `quant_config` 传给可量化 layer，例如：

```text
QKVParallelLinear(..., quant_config=quant_config)
RowParallelLinear(..., quant_config=quant_config)
MergedColumnParallelLinear(..., quant_config=quant_config)
FusedMoE(..., quant_config=quant_config)
Attention(..., quant_config=quant_config)
```

模型层面的职责是：

```text
1. 从 vllm_config.quant_config 取到 quant_config；
2. 构造 layer 时传下去；
3. 提供 packed_modules_mapping / hf_to_vllm_mapper；
4. 在 load_weights() 中配合 AutoWeightsLoader 或自定义 loader 加载权重。
```

量化方法层面的职责是：

```text
1. 判断这个 layer 是否应该量化；
2. 创建对应参数；
3. 正确加载 checkpoint tensor；
4. 后处理为 kernel-ready layout；
5. forward 时调用正确 kernel。
```

---

## 20. 和 model loader 的边界

### 20.1 model loader 负责

```text
- 找到 checkpoint 文件；
- 读取 safetensors / bin 权重；
- 根据参数名把 tensor 分发给 module parameter；
- 应用 weight_loader；
- 在加载后调用 process_weights_after_loading()。
```

### 20.2 quantization 负责

```text
- 告诉 loader 哪些参数存在；
- 定义 scale / zero point / packed weight 的参数形状；
- 定义权重加载后的重排和转换；
- 提供 checkpoint 名称到 vLLM 名称的映射；
- 提供最终 forward 的执行方法。
```

### 20.3 两者如何对接

关键接口是：

```text
Parameter.weight_loader
quant_method.create_weights()
quant_method.process_weights_after_loading()
quant_config.get_cache_scale_mapper()
```

所以 loader 本身不需要理解每种量化算法的数学细节，它只按参数和 loader 函数把 tensor 放到位。

---

## 21. 和 attention backend 的边界

### 21.1 Attention 层负责

```text
- 保存 kv_cache_dtype；
- 初始化 q/k/v/prob scale；
- 在必要时加载 checkpoint scale；
- 构造 KVCacheSpec；
- 调用具体 attention backend。
```

### 21.2 Attention backend 负责

```text
- 按 kv_cache_dtype 写入 KV cache；
- 按 scale 读取 / 反量化 KV cache；
- 执行 prefill / decode attention；
- 处理 backend 特定 layout。
```

### 21.3 量化方法负责

```text
- 提供 BaseKVCacheMethod；
- 创建可加载的 scale 参数；
- 后处理 scale 并写入 Attention 的内部 buffer。
```

这三者的边界可以压缩成：

```text
CacheConfig 决定 dtype；
QuantizationConfig 决定 scale 如何加载；
Attention backend 决定如何执行。
```

---

## 22. 容易混淆的点

### 22.1 `--quantization` 和 `--kv-cache-dtype` 是一回事吗？

不是。

```text
--quantization：
  主要控制权重 / 激活 / MoE 等量化方法。

--kv-cache-dtype：
  控制 KV cache 的存储 dtype。
```

它们可以同时使用，也可以只使用其中一个。

### 22.2 有 quantization_config 就一定是 checkpoint 量化吗？

不是。

```text
HF config 里的 quantization_config：
  通常描述 checkpoint 已经如何量化。

用户传入的 --quantization-config：
  在 vLLM 中主要用于 online quantization / activation override。
```

### 22.3 量化是不是只影响 Linear？

不是。

Linear 是最常见入口，但量化还可能影响：

```text
Embedding / LM head
MoE experts
Attention KV cache
特殊模型自定义层
```

### 22.4 KV cache 量化是不是 checkpoint 权重量化的一部分？

不完全是。

KV cache 是 runtime 中产生的 key/value 缓存，不是模型静态权重。

checkpoint 可能只提供 KV scale，但真正的 KV cache 内容是在推理过程中写入的。

### 22.5 `process_weights_after_loading()` 是不是可选优化？

不是。

对很多量化方法来说，它是从 checkpoint layout 转成 kernel layout 的必要步骤。

没有这一步，权重可能形状对了，但 kernel 不能正确使用。

### 22.6 量化方法是不是直接决定 GPU kernel？

大多数情况下是间接决定。

```text
quant_method.apply()
  → 根据量化方法、平台、dtype、shape、backend
  → 调用具体 custom op / Triton / CUTLASS / Marlin / MoE kernel。
```

### 22.7 在线量化和预量化 checkpoint 有什么区别？

```text
预量化 checkpoint：
  权重已经以量化格式保存，vLLM 负责读取、映射、repack、执行。

在线量化：
  checkpoint 可能是 fp16/bf16，vLLM 在加载后或加载过程中量化成低 bit 格式。
```

---

## 23. 最小心智模型

如果只记一条主线，可以记：

```text
QuantizationConfig 决定“这个 layer 用哪种量化方法”，
QuantizeMethod 决定“这个 layer 的参数怎么建、权重怎么处理、forward 怎么跑”。
```

更完整一点是：

```text
配置层：
  解析用户 / checkpoint 的量化意图。

模型构造层：
  把 quant_config 传给 Linear / MoE / Attention。

参数层：
  quant_method.create_weights() 注册低 bit 权重和 scale。

加载层：
  loader 加载 checkpoint tensor，process_weights_after_loading() 转成 kernel layout。

执行层：
  quant_method.apply() 或 attention backend 调用低 bit kernel。
```

再压缩成一句话：

```text
量化是 vLLM 把“checkpoint 的低 bit 表示”接到“runtime 高性能 kernel”上的适配层。
```

---

## 24. 总结

`Quantization` 在 vLLM 中的职责可以归纳为：

```text
1. 识别量化方法：从 CLI、HF config、compression_config 或在线量化配置中确定方法；
2. 校验可执行性：检查平台 capability、activation dtype、deprecated 方法；
3. 接入 layer：为 Linear / MoE / Attention 返回合适 quant_method；
4. 创建参数：注册 qweight、scale、zero point、KV scale 等量化相关参数；
5. 加载权重：配合 loader 把 checkpoint tensor 映射到 vLLM 参数；
6. 后处理权重：repack / transpose / online quantize / scale 整理；
7. 执行 kernel：在 forward 中调用对应低 bit GEMM / MoE / attention kernel；
8. 管理 KV cache scale：处理 fp8 KV cache 的 q/k/v/prob scale。
```

如果只记一句话：

```text
vLLM 的量化不是一个模型开关，而是配置、loader、layer、parameter、attention cache 和 kernel 之间的一套协议。
```
