# 09. MoE / Fused MoE 量化如何处理？

源码位置：

- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\routed_experts.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\fused_moe_method_base.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\fused_moe_modular_method.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\config.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\runner\moe_runner.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\router\gate_linear.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\quantization\fp8.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\quantization\auto_awq.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\quantization\auto_gptq.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\quantization\moe_wna16.py`

本问题关注：Mixture-of-Experts 模型中 routed experts 的 `w1 / w2 / w3` 权重如何被组织成 fused MoE 参数，量化方法如何创建 per-expert 的低 bit 权重、scale、zero point、g_idx，以及 fused MoE kernel 如何消费这些量化参数。

---

## 1. 一句话回答

MoE 量化不是简单把每个 expert 当成普通 Linear 逐个量化，而是把所有本地 expert 的权重组织成一组 fused expert tensors，再交给 `FusedMoEMethodBase` 派生的量化方法处理。

主链路是：

```text
MoE layer 构造 router / routed experts
  → Router/GateLinear 计算 router_logits
  → RoutedExperts 根据 quant_config 选择 FusedMoEMethodBase
  → quant_method.create_weights() 创建 w13 / w2 或 qweight / scales / qzeros
  → RoutedExperts.weight_loader() 按 expert_id + shard_id(w1/w2/w3) 加载权重
  → quant_method.process_weights_after_loading() 转换成 kernel 运行时布局
  → MoERunner 根据 router_logits 得到 topk_weights/topk_ids
  → RoutedExperts.forward_modular() 或 forward_monolithic()
  → quant_method.apply() / apply_monolithic()
  → FusedMoEKernel / fused_experts 执行 routed expert GEMM 和结果合并
```

所以：

```text
普通 Linear 量化关注单个矩阵；
MoE 量化关注 expert 维度、w1/w3 融合、w2 单独存储、routing/top-k、EP/TP 切分以及 fused kernel 需要的布局。
```

---

## 2. 最小心智模型

一个典型 gated MoE expert 有三组权重：

```text
w1 / gate_proj：hidden → intermediate
w3 / up_proj：hidden → intermediate
w2 / down_proj：intermediate → hidden
```

vLLM 在 `RoutedExperts` 中通常把它们组织成：

```text
w13：把 w1 和 w3 沿 intermediate/output 维度融合
w2：down projection 单独保存
```

未量化时是：

```text
w13_weight: [num_local_experts, 2 * intermediate_size_per_partition, hidden_size]
w2_weight:  [num_local_experts, hidden_size, intermediate_size_per_partition]
```

量化后可能变成：

```text
FP8：
  w13_weight / w2_weight
  w13_weight_scale / w2_weight_scale
  w13_input_scale / w2_input_scale

AWQ / GPTQ / WNA16：
  w13_qweight / w2_qweight
  w13_scales / w2_scales
  w13_qzeros / w2_qzeros
  w13_g_idx / w2_g_idx  # GPTQ desc_act 场景
```

但执行语义仍然是：

```text
hidden states
  → router logits
  → top-k experts
  → 每个 token 路由到若干 expert
  → expert GEMM + activation + down GEMM
  → 按 top-k weight 合并输出
```

---

## 3. Router/Gate 通常不是 MoE 量化的重点

Router/gate 负责把 hidden states 投影成 expert logits。

源码：`gate_linear.py`

`GateLinear` 继承 `ReplicatedLinear`，但构造时显式传入：

```python
quant_config=None
```

位置：`gate_linear.py:58` 到 `gate_linear.py:65`

这说明：

```text
MoE expert 权重量化，不等于 router/gate 也量化。
```

`GateLinear` 更关注 router logits 的计算性能和 dtype，例如：

```text
DSV3 specialized router GEMM；
fp32 specialized router GEMM；
cuBLAS bf16×bf16→fp32；
普通 ReplicatedLinear fallback。
```

位置：`gate_linear.py:112` 到 `gate_linear.py:145`

所以本文说的 MoE 量化主要指：

```text
RoutedExperts 内部的 expert weights。
```

不是 router logits 这层。

---

## 4. RoutedExperts 是 MoE 量化的承载层

入口在：`routed_experts.py:43`

`RoutedExperts` 的定位是：

```text
保存本 rank / 本 expert partition 上的 expert 权重；
提供 MoE 专用 weight_loader；
根据 quant_method 执行 routed experts。
```

类注释也直接说明：

```text
This module owns the expert weight parameters (w13_weight, w2_weight, scales, etc.)
and handles:
- Loading checkpoint weights into parameters
- Executing routed experts via quant_method.apply()
```

位置：`routed_experts.py:44` 到 `routed_experts.py:52`

---

## 5. RoutedExperts 如何选择 quant_method

初始化时：

```python
self.quant_method = self._get_quant_method(
    self.layer_name,
    self.quant_config,
    self.moe_config,
)
```

位置：`routed_experts.py:114` 到 `routed_experts.py:118`

选择逻辑：

```python
quant_method = None
if quant_config is not None:
    quant_method = quant_config.get_quant_method(self, prefix)
if quant_method is None:
    quant_method = UnquantizedFusedMoEMethod(moe_config)
assert isinstance(quant_method, FusedMoEMethodBase)
```

位置：`routed_experts.py:180` 到 `routed_experts.py:196`

这说明：

```text
1. MoE 不使用 LinearMethodBase，而使用 FusedMoEMethodBase；
2. 如果量化配置不支持 RoutedExperts，就回退 UnquantizedFusedMoEMethod；
3. 普通 Linear 支持某种量化，不代表 MoE 一定支持同样的量化路径。
```

---

## 6. FusedMoEMethodBase 的职责

抽象基类在：`fused_moe_method_base.py:31`

关键字段：

```python
self.moe: FusedMoEConfig
self.moe_quant_config: FusedMoEQuantConfig | None
self.moe_kernel: FusedMoEKernel | None
```

位置：`fused_moe_method_base.py:31` 到 `fused_moe_method_base.py:37`

关键方法：

```python
def create_weights(
    self,
    layer,
    num_experts,
    hidden_size,
    intermediate_size_per_partition,
    params_dtype,
    **extra_weight_attrs,
)
```

位置：`fused_moe_method_base.py:52` 到 `fused_moe_method_base.py:62`

执行接口分两类：

```python
def apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)

def apply_monolithic(layer, x, router_logits, input_ids=None)
```

位置：`fused_moe_method_base.py:172` 到 `fused_moe_method_base.py:214`

可以理解为：

```text
create_weights：创建 expert 权重和量化参数；
process_weights_after_loading：把 checkpoint 形态转成 kernel 形态；
get_fused_moe_quant_config：生成 kernel 可理解的量化描述；
apply：routing 已经算好 top-k 的 modular kernel 路径；
apply_monolithic：kernel 内部自己处理 routing 的 monolithic 路径。
```

---

## 7. FusedMoEConfig 和 parallel config

`FusedMoEConfig` 在：`config.py:1258`

它记录 MoE 层的核心结构：

```text
num_experts：全局 expert 数；
experts_per_token：每个 token 选几个 expert；
hidden_dim；
intermediate_size；
num_local_experts；
num_logical_experts；
activation；
routing_method；
moe_parallel_config；
in_dtype；
max_num_tokens；
has_bias；
```

位置：`config.py:1258` 到 `config.py:1297`

`__post_init__()` 中会计算：

```python
self.intermediate_size_per_partition = self.intermediate_size // tp_size
```

位置：`config.py:1300` 到 `config.py:1305`

因此 MoE quant method 创建参数时拿到的是：

```text
hidden_size；
intermediate_size_per_partition；
num_local_experts；
params_dtype；
weight_loader。
```

---

## 8. TP / EP 对 MoE 权重量化的影响

MoE 的并行配置在 `FusedMoEParallelConfig`。

位置：`config.py:1017`

### 8.1 不启用 EP

如果不启用 expert parallel，可能会把 TP 维度跨 DP / PCP flatten：

```python
tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(...)
```

位置：`config.py:1197` 到 `config.py:1199`

然后：

```text
expert 不切分到不同设备；
权重按 TP 切 intermediate 维度；
w13 类似 MergedColumnParallel；
w2 类似 RowParallel。
```

### 8.2 启用 EP

启用 EP 时：

```python
ep_size = tp_size
ep_rank = tp_rank
return FusedMoEParallelConfig(tp_size=1, ep_size=ep_size, ...)
```

位置：`config.py:1216` 到 `config.py:1235`

含义是：

```text
每个设备拥有一部分完整 expert；
expert_map 决定 global expert id 到 local expert id 的映射；
TP 维度被折叠为 1，专家在 EP 维度切分。
```

因此 MoE 量化权重加载必须同时考虑：

```text
expert_id 是否属于本 rank；
TP rank 对 w13/w2 的切片；
EP 下 global expert → local expert 的映射；
EPLB 下 redundant expert / physical expert id。
```

---

## 9. RoutedExperts.create_weights 调用

`RoutedExperts.__init__()` 先准备 `moe_quant_params`：

```python
moe_quant_params = {
    "num_experts": moe_config.num_local_experts,
    "hidden_size": self.hidden_size,
    "unpadded_hidden_size": self.moe_config.hidden_dim_unpadded,
    "intermediate_size_per_partition": self.moe_config.intermediate_size_per_partition,
    "params_dtype": params_dtype,
    "weight_loader": self.weight_loader,
    "global_num_experts": moe_config.num_experts,
}
```

位置：`routed_experts.py:150` 到 `routed_experts.py:160`

然后调用：

```python
self.quant_method.create_weights(layer=self, **moe_quant_params)
```

位置：`routed_experts.py:168`

这和普通 Linear 的模式一致：

```text
Layer 负责提供尺寸、loader、prefix/context；
quant_method 负责注册具体参数。
```

但 MoE 多了两个维度：

```text
expert 维度；
w1/w2/w3 shard 维度。
```

---

## 10. 未量化 MoE 的参数布局

未量化实现：`unquantized_fused_moe_method.py`

### 10.1 create_weights

如果是 gated activation，`w13` 宽度是 `2 * intermediate_size_per_partition`：

```python
w13_weight = torch.nn.Parameter(
    torch.empty(
        num_experts,
        2 * intermediate_size_per_partition,
        hidden_size,
        dtype=params_dtype,
    ),
    requires_grad=False,
)
```

位置：`unquantized_fused_moe_method.py:97` 到 `unquantized_fused_moe_method.py:112`

`w2` 是：

```python
w2_weight = torch.nn.Parameter(
    torch.empty(
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        dtype=params_dtype,
    ),
    requires_grad=False,
)
```

位置：`unquantized_fused_moe_method.py:120` 到 `unquantized_fused_moe_method.py:131`

如果 `moe.has_bias`，还会创建：

```text
w13_bias: [num_experts, w13_up_dim]
w2_bias:  [num_experts, hidden_size]
```

位置：`unquantized_fused_moe_method.py:113` 到 `unquantized_fused_moe_method.py:138`

### 10.2 process_weights_after_loading

加载后会调用：

```python
convert_to_unquantized_kernel_format(...)
replace_parameter(layer, "w13_weight", w13_new)
replace_parameter(layer, "w2_weight", w2_new)
make_unquantized_moe_kernel(...)
```

位置：`unquantized_fused_moe_method.py:155` 到 `unquantized_fused_moe_method.py:199`

这说明即使未量化，MoE 也可能需要把 checkpoint 权重转换成 fused kernel 的运行时布局。

---

## 11. RoutedExperts.weight_loader 如何理解 w1/w2/w3

MoE checkpoint 通常按 expert 分开存：

```text
experts.0.gate_proj.weight  → w1
experts.0.down_proj.weight  → w2
experts.0.up_proj.weight    → w3
```

`RoutedExperts.weight_loader()` 接收：

```python
param,
loaded_weight,
weight_name,
shard_id,
expert_id,
```

位置：`routed_experts.py:571` 到 `routed_experts.py:579`

其中：

```text
shard_id：w1 / w2 / w3；
expert_id：checkpoint 中的 expert id；
weight_name：用于判断是 weight / scale / zero / g_idx / input_scale。
```

加载前会先映射 expert：

```python
global_expert_id = expert_id
expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)
```

位置：`routed_experts.py:592` 到 `routed_experts.py:595`

如果该 expert 不属于本 rank：

```python
if expert_id == -1 and not use_global_sf:
    return False
```

位置：`routed_experts.py:596` 到 `routed_experts.py:604`

所以 EP 下权重加载不是所有 rank 都 copy 全量 experts，而是：

```text
只加载本 rank 拥有的 local experts；
global scale 等少数参数可能例外。
```

---

## 12. w13 和 w2 的加载切片规则

`RoutedExperts` 把 `w1/w3` 加到 `w13`，把 `w2` 加到 `w2`。

### 12.1 w13 加载

`_load_w13()` 中：

```python
if shard_id == "w1":
    expert_data = expert_data.narrow(shard_dim, 0, shard_size)
else:
    expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
```

位置：`routed_experts.py:436` 到 `routed_experts.py:485`

含义是：

```text
w1 放进 w13 前半段；
w3 放进 w13 后半段。
```

### 12.2 w2 加载

`_load_w2()` 中：

```text
down_proj 按 RowParallel 语义在 intermediate/input 维度切片。
```

位置：`routed_experts.py:487` 到 `routed_experts.py:520`

### 12.3 padding 处理

如果 kernel 因 DeepEP / MXFP4 等要求把 hidden 或 intermediate round up，加载时会 narrow 到 checkpoint 原始大小：

```python
expert_data = self._narrow_expert_data_for_padding(...)
```

位置：`routed_experts.py:402` 到 `routed_experts.py:434`

这保证：

```text
参数可以为 kernel 对齐而更大；
checkpoint 仍按原始形状加载到有效区域。
```

---

## 13. scale / zero point / g_idx 的加载规则

`weight_loader()` 根据 `weight_name` 分支处理量化辅助参数。

### 13.1 input_scale

```python
if "input_scale" in weight_name:
    self._load_single_value(...)
```

位置：`routed_experts.py:680` 到 `routed_experts.py:714`

某些 compressed/modelopt 场景会校验 w1/w3 input scale 是否一致，或按 global scale 加载。

### 13.2 g_idx

```python
if "g_idx" in weight_name:
    self._load_g_idx(...)
```

位置：`routed_experts.py:716` 到 `routed_experts.py:725`

GPTQ desc_act / activation order 场景会用到。

### 13.3 scale / zero / offset

```python
if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
    quant_method = getattr(param, "quant_method", None)
```

位置：`routed_experts.py:790` 到 `routed_experts.py:830`

支持的 scale 类型由 `FusedMoeWeightScaleSupported` 表示：

```text
TENSOR：per expert / per logical tensor；
CHANNEL：per output channel；
GROUP：按 group_size 分组；
BLOCK：按 block_n/block_k 分块。
```

定义位置：`routed_experts.py:36` 到 `routed_experts.py:40`

---

## 14. FusedMoEQuantConfig 是 kernel 的量化说明书

`FusedMoEQuantConfig` 在：`config.py:214`

它包含四个 `FusedMoEQuantDesc`：

```text
_a1：第一段 GEMM 的 activation 量化描述；
_a2：第二段 GEMM 的 activation 量化描述；
_w1：w13 / gate_up 的权重量化描述；
_w2：down_proj 的权重量化描述。
```

位置：`config.py:248` 到 `config.py:253`

每个 `FusedMoEQuantDesc` 可以描述：

```text
dtype：量化 dtype，例如 fp8/int8/int4/mxfp4/nvfp4；
shape：per tensor / per token / group / block；
scale：scale tensor 或 PrecisionConfig；
alpha_or_gscale：global scale / alpha；
zp：zero point；
bias：bias。
```

位置：`config.py:174` 到 `config.py:209`

它的作用是：

```text
把不同 quant_method 的参数命名和布局，统一翻译成 fused MoE kernel 能理解的配置。
```

---

## 15. FusedMoEQuantConfig 支持哪些量化形态

`config.py` 提供了一批 builder：

```text
fp8_w8a8_moe_quant_config：FP8 activation + FP8 weight；
fp8_w8a16_moe_quant_config：FP16/BF16 activation + FP8 weight；
int8_w8a8_moe_quant_config：INT8 activation + INT8 weight；
int8_w8a16_moe_quant_config：FP16/BF16 activation + INT8 weight；
int4_w4a16_moe_quant_config：FP16/BF16 activation + INT4 weight；
gptq_marlin_moe_quant_config：GPTQ/Marlin W4/W8 A16；
mxfp4_w4a16_moe_quant_config；
nvfp4_moe_quant_config；
ocp_mx_moe_quant_config。
```

典型位置：

```text
FP8 W8A8：config.py:592
gptq_marlin：config.py:660
MXFP4 W4A16：config.py:696
INT4 W4A16：config.py:878
FP8 W8A16：config.py:901
INT8 W8A16：config.py:941
```

所以 MoE quant method 最终要做的一件事是：

```text
基于 layer 上已经加载/转换好的参数，构造 FusedMoEQuantConfig。
```

---

## 16. MoERunner 的执行链路

`MoERunner.forward()` 在：`moe_runner.py:628`

主输入是：

```text
hidden_states
router_logits
input_ids（可选）
```

位置：`moe_runner.py:628` 到 `moe_runner.py:633`

执行大致是：

```text
hidden_states
  → apply_routed_input_transform()
  → _maybe_pad_hidden_states()
  → _forward_entry(...)
  → shared_output / fused_output
  → 必要时裁剪 padding
  → shared experts 合并
  → 必要时 all-reduce
  → output hidden_states
```

位置：`moe_runner.py:654` 到 `moe_runner.py:681`

对于 modular kernel，内部会先算 top-k：

```python
topk_weights, topk_ids = self.router.select_experts(
    hidden_states=hidden_states,
    router_logits=router_logits,
    topk_indices_dtype=self._quant_method.topk_indices_dtype,
    input_ids=input_ids,
)
```

位置：`moe_runner.py:560` 到 `moe_runner.py:565`

然后执行 routed experts：

```python
fused_out = self.routed_experts.forward_modular(
    x=hidden_states,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    shared_experts=self._shared_experts,
    shared_experts_input=shared_experts_input,
)
```

位置：`moe_runner.py:567` 到 `moe_runner.py:573`

---

## 17. modular 和 monolithic 两种执行路径

`RoutedExperts` 不允许直接调用 `forward()`：

```python
def forward(...):
    raise AssertionError("Call forward_modular or forward_monolithic instead.")
```

位置：`routed_experts.py:1122` 到 `routed_experts.py:1127`

### 17.1 modular 路径

```python
return self.quant_method.apply(
    layer=self,
    x=x,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    shared_experts=shared_experts,
    shared_experts_input=shared_experts_input,
)
```

位置：`routed_experts.py:1053` 到 `routed_experts.py:1088`

特点：

```text
router 已经算好 topk_weights / topk_ids；
prepare/finalize 和 experts GEMM 可以拆成模块；
更适合 all2all、EP、shared experts overlap 等路径。
```

### 17.2 monolithic 路径

```python
return self.quant_method.apply_monolithic(
    layer=self,
    x=x,
    router_logits=router_logits,
    input_ids=input_ids,
)
```

位置：`routed_experts.py:1090` 到 `routed_experts.py:1120`

特点：

```text
kernel 内部处理 routing；
CPU 或部分后端仍可能使用；
接口直接吃 router_logits。
```

---

## 18. FusedMoEModularMethod 做什么

`FusedMoEModularMethod` 在：`fused_moe_modular_method.py:32`

它是一个包装器，用已有 quant_method 和 `FusedMoEKernel` 组成 modular 方法：

```python
return FusedMoEModularMethod(
    old_quant_method,
    FusedMoEKernel(
        prepare_finalize,
        old_quant_method.select_gemm_impl(prepare_finalize, routed_experts),
    ),
)
```

位置：`fused_moe_modular_method.py:51` 到 `fused_moe_modular_method.py:62`

forward 时统一调用：

```python
return self.moe_kernel.apply(
    hidden_states=x,
    w1=layer.w13_weight,
    w2=layer.w2_weight,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    activation=layer.activation,
    global_num_experts=layer.global_num_experts,
    apply_router_weight_on_input=layer.apply_router_weight_on_input,
    expert_map=layer.expert_map,
    shared_experts=shared_experts,
    shared_experts_input=shared_experts_input,
)
```

位置：`fused_moe_modular_method.py:96` 到 `fused_moe_modular_method.py:118`

这说明 modular kernel 的核心输入就是：

```text
hidden states；
w13/w2 expert weights；
top-k routing 结果；
activation 类型；
expert_map；
shared experts。
```

---

## 19. FP8 MoE 量化

FP8 实现在：`fp8.py`

### 19.1 get_quant_method

`Fp8Config.get_quant_method()` 对 `RoutedExperts` 的分支是：

```python
elif isinstance(layer, RoutedExperts):
    if is_layer_skipped(...):
        return UnquantizedFusedMoEMethod(layer.moe_config)
    if self.store_dtype == "mxfp4":
        return Mxfp4MoEMethod(layer.moe_config)
    if self.is_checkpoint_fp8_serialized:
        moe_quant_method = Fp8MoEMethod(self, layer)
    else:
        moe_quant_method = Fp8OnlineMoEMethod(self, layer)
    return moe_quant_method
```

位置：`fp8.py:201` 到 `fp8.py:218`

含义：

```text
FP8 serialized checkpoint：加载 FP8 expert 权重；
非 FP8 checkpoint：在线把 BF16/FP16 expert 权重量化成 FP8；
ignored layer：回退未量化 MoE；
store_dtype=mxfp4：走 MXFP4 MoE 方法。
```

### 19.2 create_weights

`Fp8MoEMethod.create_weights()` 创建：

```text
w13_weight: FP8, [E, 2I, H]
w2_weight:  FP8, [E, H, I]
w13_weight_scale / w2_weight_scale
或 w13_weight_scale_inv / w2_weight_scale_inv
可选 w13_input_scale / w2_input_scale
可选 bias
```

位置：`fp8.py:534` 到 `fp8.py:672`

per-tensor scale：

```text
w13_scale_data: [num_experts, 2]
w2_scale_data:  [num_experts]
```

位置：`fp8.py:620` 到 `fp8.py:624`

block FP8 scale：

```text
w13_scale_data: [E, 2 * ceil(I / block_n), ceil(H / block_k)]
w2_scale_data:  [E, ceil(H / block_n), ceil(I / block_k)]
```

位置：`fp8.py:625` 到 `fp8.py:638`

### 19.3 process_weights_after_loading

FP8 加载后会：

```text
1. 在 MI300/MI325 这类平台上转换 FP8 FNUZ；
2. 静态 activation scale 场景下合并 input scales；
3. per-tensor w13 场景下处理 w1/w3 两个 scale；
4. convert_to_fp8_moe_kernel_format() 转为 backend 运行时布局；
5. replace_parameter() 更新 layer 权重和 scale；
6. make_fp8_moe_kernel() 创建 FusedMoEKernel。
```

位置：`fp8.py:719` 到 `fp8.py:762`

核心 setup：

```python
w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(...)
...
self.moe_kernel = make_fp8_moe_kernel(...)
```

位置：`fp8.py:674` 到 `fp8.py:717`

### 19.4 apply

modular 路径：

```python
return self.moe_kernel.apply(
    x,
    layer.w13_weight,
    layer.w2_weight,
    topk_weights,
    topk_ids,
    ...
)
```

位置：`fp8.py:831` 到 `fp8.py:854`

monolithic 路径：

```python
return self.moe_kernel.apply_monolithic(...)
```

位置：`fp8.py:807` 到 `fp8.py:829`

---

## 20. FP8 MoE backend 选择

FP8 MoE backend 选择在：`fused_moe/oracle/fp8.py`

支持的 backend enum 包括：

```text
FLASHINFER_TRTLLM
FLASHINFER_CUTLASS
DEEPGEMM
BATCHED_DEEPGEMM
MARLIN
TRITON
BATCHED_TRITON
AITER
VLLM_CUTLASS
BATCHED_VLLM_CUTLASS
XPU
CPU
EMULATION
NATIVE_MXFP8
```

位置：`oracle/fp8.py:41` 到 `oracle/fp8.py:61`

选择时会根据：

```text
平台：CUDA / ROCm / XPU / CPU；
GPU capability：Hopper / Blackwell 等；
是否 EP / DeepEP；
weight_key / activation_key；
moe_backend 用户指定值；
kernel 对 shape/quant config 的支持情况。
```

初始优先级在：`oracle/fp8.py:64` 到 `oracle/fp8.py:126`

backend 到 experts kernel 类的映射在：`oracle/fp8.py:129` 到 `oracle/fp8.py:220`

这解释了为什么同样是 FP8 MoE，不同机器上可能走不同 kernel。

---

## 21. AWQ MoE 量化

AWQ MoE 在：`auto_awq.py:544`

### 21.1 选择条件

`AutoAWQConfig.get_quant_method()` 中，如果 layer 是 `RoutedExperts`：

```python
elif isinstance(layer, RoutedExperts):
    if is_layer_skipped(...):
        return UnquantizedFusedMoEMethod(layer.moe_config)

    if not check_moe_marlin_supports_layer(layer, self.group_size):
        return MoeWNA16Config.from_config(self.full_config).get_quant_method(layer, prefix)

    return AutoAWQMoEMethod(self, layer.moe_config)
```

位置：`auto_awq.py:334` 到 `auto_awq.py:355`

含义：

```text
AWQ MoE 优先走 AutoAWQMoEMethod；
如果 MoE Marlin 不支持当前 layer，则 fallback 到 MoeWNA16Config；
跳过模块则用 UnquantizedFusedMoEMethod。
```

### 21.2 create_weights

AWQ MoE 创建：

```text
w13_qweight: [E, H, 2I / pack_factor]
w2_qweight:  [E, I, H / pack_factor]
w13_scales:  [E, num_groups_w13, 2I]
w2_scales:   [E, num_groups_w2, H]
w13_qzeros:  [E, num_groups_w13, 2I / pack_factor]
w2_qzeros:   [E, num_groups_w2, H / pack_factor]
workspace
```

位置：`auto_awq.py:562` 到 `auto_awq.py:661`

注意它会设置：

```python
extra_weight_attrs.update({
    "is_transposed": True,
    "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
})
```

位置：`auto_awq.py:571` 到 `auto_awq.py:577`

这会影响 `RoutedExperts.weight_loader()` 如何理解 shard 维度和 scale 类型。

### 21.3 process_weights_after_loading

加载后会调用：

```python
convert_to_wna16_moe_kernel_format(...)
replace_parameter(layer, "w13_qweight", w13)
replace_parameter(layer, "w2_qweight", w2)
...
self._setup_kernel(layer)
```

位置：`auto_awq.py:663` 到 `auto_awq.py:725`

并且会把：

```python
layer.w13_weight = layer.w13_qweight
layer.w2_weight = layer.w2_qweight
```

位置：`auto_awq.py:697` 到 `auto_awq.py:702`

原因是 modular kernel 的统一接口期待 `w13_weight/w2_weight` 名称，但 AWQ 内部真实参数叫 `qweight`。

### 21.4 apply

```python
return self.moe_kernel.apply(
    hidden_states=x,
    w1=layer.w13_qweight,
    w2=layer.w2_qweight,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    ...
)
```

位置：`auto_awq.py:771` 到 `auto_awq.py:794`

---

## 22. GPTQ MoE 量化

GPTQ MoE 在：`auto_gptq.py:465`

### 22.1 选择条件

`AutoGPTQConfig.get_quant_method()` 中：

```python
if isinstance(layer, RoutedExperts):
    if not check_moe_marlin_supports_layer(layer, self.group_size):
        return MoeWNA16Config.from_config(self.full_config).get_quant_method(layer, prefix)
    moe_quant_method = get_moe_quant_method(self, layer, prefix, AutoGPTQMoEMethod)
    moe_quant_method.input_dtype = get_marlin_input_dtype(prefix)
    return moe_quant_method
```

位置：`auto_gptq.py:240` 到 `auto_gptq.py:260`

GPTQ 支持 dynamic 配置，可能按 regex 跳过某些 MoE 层。

位置：`auto_gptq.py:121` 到 `auto_gptq.py:145`

### 22.2 create_weights

GPTQ MoE 创建：

```text
w13_qweight: [E, H / pack_factor, 2I]
w2_qweight:  [E, I / pack_factor, H]
w13_scales:  [E, scales_size13, 2I]
w2_scales:   [E, scales_size2, H]
w13_qzeros / w2_qzeros
w13_g_idx / w2_g_idx
w13_g_idx_sort_indices / w2_g_idx_sort_indices
workspace（modular experts 时）
```

位置：`auto_gptq.py:492` 到 `auto_gptq.py:650`

它也设置：

```python
extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})
```

位置：`auto_gptq.py:532`

### 22.3 desc_act 对 w2 scale 的影响

GPTQ 如果启用 `desc_act`：

```python
w2_scales_size = intermediate_size_full if self.quant_config.desc_act else intermediate_size_per_partition
...
set_weight_attrs(w2_scales, {"load_full_w2": self.quant_config.desc_act})
set_weight_attrs(w2_qzeros, {"load_full_w2": self.quant_config.desc_act})
```

位置：`auto_gptq.py:515` 到 `auto_gptq.py:577`

含义：

```text
activation order 场景下，w2 的 scales/qzeros 可能需要加载完整 intermediate 维度，而不是只加载 TP shard。
```

### 22.4 process 和 apply

加载后同样通过：

```python
convert_to_wna16_moe_kernel_format(...)
make_wna16_moe_kernel(...)
```

位置：`auto_gptq.py:651` 到 `auto_gptq.py:750`

forward 时：

```python
return self.moe_kernel.apply(
    hidden_states=x,
    w1=layer.w13_qweight,
    w2=layer.w2_qweight,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    ...
)
```

位置：`auto_gptq.py:783` 到 `auto_gptq.py:806`

---

## 23. MoeWNA16 是什么

`moe_wna16.py` 提供 W8A16 / W4A16 MoE 量化兼容路径。

位置：`moe_wna16.py:34`

它支持的来源是：

```text
GPTQ：int4 / int8，通常要求 desc_act=False 才能兼容基础判断；
AWQ：int4。
```

兼容性判断：`moe_wna16.py:133` 到 `moe_wna16.py:156`

### 23.1 get_quant_method

对不同 layer：

```python
if skipped:
    RoutedExperts → UnquantizedFusedMoEMethod
    LinearBase → UnquantizedLinearMethod
elif isinstance(layer, LinearBase):
    转给 AutoGPTQConfig 或 AutoAWQConfig
elif isinstance(layer, RoutedExperts):
    return MoeWNA16Method(self, layer.moe_config)
```

位置：`moe_wna16.py:158` 到 `moe_wna16.py:184`

这说明 `moe_wna16` 不是只管 MoE，它也会把普通 Linear 继续委托给对应的 AWQ/GPTQ 实现。

### 23.2 create_weights

`MoeWNA16Method.create_weights()` 创建 uint8 packed 参数：

```text
w13_qweight: [E, 2I, H / bit8_pack_factor]
w2_qweight:  [E, H, I / bit8_pack_factor]
w13_scales:  [E, 2I, H / group_size]
w2_scales:   [E, H, I / group_size]
w13_qzeros / w2_qzeros（如果 has_zp）
```

位置：`moe_wna16.py:202` 到 `moe_wna16.py:320`

它还会在 hidden/intermediate 不能整除 group_size 时逐步缩小 group size：

```python
while intermediate_size_per_partition % group_size or hidden_size % group_size:
    group_size = group_size // 2
    group_size_div_factor *= 2
    assert group_size >= 32
```

位置：`moe_wna16.py:216` 到 `moe_wna16.py:224`

### 23.3 自定义 weight_loader

`MoeWNA16Method.get_weight_loader()` 会包装原始 loader。

位置：`moe_wna16.py:366` 到 `moe_wna16.py:490`

它会把 AWQ/GPTQ checkpoint 转成统一格式：

```text
AWQ：
  修正 int4 pack order；
  qweight/qzeros 转置并 repack 成 uint8。

GPTQ：
  weight 转置后 view(uint8)；
  qzeros 转成 uint8，并加 1 对齐 AWQ 表示。
```

位置：`moe_wna16.py:368` 到 `moe_wna16.py:455`

如果 group size 被缩小，还会重复 scale/qzeros：

```python
loaded_weight = loaded_weight.repeat_interleave(layer.group_size_div_factor, 1)
```

位置：`moe_wna16.py:457` 到 `moe_wna16.py:465`

### 23.4 apply

`MoeWNA16Method.apply()` 不创建 `FusedMoEKernel`，而是直接调用 `fused_experts`：

```python
return fused_experts(
    x,
    layer.w13_qweight,
    layer.w2_qweight,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    activation=layer.activation,
    global_num_experts=layer.global_num_experts,
    expert_map=layer.expert_map,
    quant_config=self.moe_quant_config,
)
```

位置：`moe_wna16.py:342` 到 `moe_wna16.py:364`

---

## 24. Fused MoE kernel 如何消费量化参数

不管是 FP8、AWQ、GPTQ，最终 kernel 大多需要：

```text
hidden_states；
w1/w13 权重；
w2 权重；
topk_weights；
topk_ids；
activation；
global_num_experts；
expert_map；
quant_config；
shared_experts（可选）。
```

典型调用：

```python
self.moe_kernel.apply(
    hidden_states=x,
    w1=layer.w13_weight or layer.w13_qweight,
    w2=layer.w2_weight or layer.w2_qweight,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    activation=layer.activation,
    global_num_experts=layer.global_num_experts,
    expert_map=layer.expert_map,
    shared_experts=shared_experts,
    shared_experts_input=shared_experts_input,
)
```

量化参数通常通过两种方式进入 kernel：

```text
1. 已经被打包进 weight tensor 的运行时布局；
2. 通过 FusedMoEQuantConfig 引用 scales / zero points / global scales / bias。
```

所以 MoE 量化的最终目标是：

```text
让权重 tensor 形状、scale 形状、expert_map、topk_ids 和 kernel backend 完全一致。
```

---

## 25. 哪些量化方法支持 fused MoE

从当前源码可以看到，显式支持 `RoutedExperts` 的典型方法包括：

```text
FP8：Fp8MoEMethod / Fp8OnlineMoEMethod；
AWQ：AutoAWQMoEMethod，或 fallback 到 MoeWNA16；
GPTQ：AutoGPTQMoEMethod，或 fallback 到 MoeWNA16；
MoeWNA16：W4A16 / W8A16 MoE；
CompressedTensors：多种 WNA16 / W8A8 / W4A8 / MXFP4 / NVFP4 MoE；
ModelOpt / Quark / online int8/fp8/mxfp8 等也有 MoE 分支。
```

判断某种 quantization 是否支持 MoE，不能只看普通 `LinearBase` 分支，要看：

```text
QuantizationConfig.get_quant_method(layer, prefix)
是否对 RoutedExperts 返回 FusedMoEMethodBase。
```

如果返回 `None`，`RoutedExperts` 会回退到未量化 MoE。

如果显式检查失败，可能会：

```text
fallback 到 MoeWNA16；
返回 UnquantizedFusedMoEMethod；
或直接 raise NotImplementedError / ValueError。
```

---

## 26. 为什么 MoE 量化比普通 Linear 复杂

普通 Linear 量化只需要处理：

```text
一个 weight；
一个输入 x；
可选 bias；
TP input/output 维度切分。
```

MoE 量化还要处理：

```text
多个 experts；
w1/w3 fused 成 w13；
w2 单独按 row-parallel 语义切；
每个 token 的 top-k expert；
expert_map / EP；
shared experts；
per expert scale；
per tensor / per channel / group / block scales；
zero points；
g_idx / desc_act；
kernel backend 对布局的要求；
process_weights_after_loading 中的 repack / shuffle / alias。
```

所以 MoE 量化真正复杂的地方不是“低 bit GEMM”本身，而是：

```text
checkpoint 权重格式 → vLLM RoutedExperts 参数格式 → fused kernel runtime 格式
```

这三者之间的转换。

---

## 27. 容易疑惑的点

### 27.1 普通 Linear 支持 AWQ/GPTQ，MoE 就一定支持吗？

不一定。

MoE 必须有 `RoutedExperts` 分支，并且 kernel 支持当前 shape / group_size / backend。

否则可能 fallback 到 `MoeWNA16` 或未量化 MoE。

### 27.2 Router/gate 是否一起量化？

通常不是本文的 MoE 量化重点。

`GateLinear` 构造时 `quant_config=None`，更多关注 router logits 的精度和 specialized GEMM。

### 27.3 w13 是什么？

`w13` 是 `w1/gate_proj` 和 `w3/up_proj` 的 fused 权重。

```text
w13 前半段：w1；
w13 后半段：w3。
```

### 27.4 为什么有 w13_qweight 但 kernel 又要 w13_weight？

部分 quant method 内部参数名是 `w13_qweight`，但 modular kernel 统一接口使用 `w1/w2` 或 `w13_weight/w2_weight` 语义。

所以 AWQ 会在加载后设置：

```python
layer.w13_weight = layer.w13_qweight
layer.w2_weight = layer.w2_qweight
```

### 27.5 expert_id 为什么会变？

checkpoint 中的是 global expert id。

EP / EPLB 下当前 rank 只拥有部分 local / physical experts，所以加载时要做：

```text
global expert id → local expert id
```

如果不属于本 rank，就跳过加载。

### 27.6 scale 为什么有 TENSOR / CHANNEL / GROUP / BLOCK？

不同量化格式的 scale 粒度不同：

```text
TENSOR：每个 expert 或每个 logical weight 一个 scale；
CHANNEL：按输出 channel；
GROUP：按 group_size；
BLOCK：按二维 block，例如 128x128。
```

kernel 需要知道这些 scale 如何广播到 GEMM。

### 27.7 process_weights_after_loading 为什么这么重要？

因为 checkpoint 格式通常不是 kernel 最想要的格式。

这一步会做：

```text
repack；
transpose；
scale 合并；
backend-specific shuffle；
Parameter 替换；
FusedMoEKernel 初始化。
```

---

## 28. 总结

MoE / Fused MoE 量化主链路可以压缩成：

```text
GateLinear 计算 router_logits
  → MoERunner / router 选择 top-k experts
  → RoutedExperts 持有本地 expert 权重
  → FusedMoEMethodBase.create_weights() 创建 w13/w2 或 qweight/scales/qzeros
  → RoutedExperts.weight_loader() 按 expert_id + w1/w2/w3 加载 checkpoint
  → process_weights_after_loading() 转为 kernel runtime layout
  → FusedMoEQuantConfig 描述 activation/weight scale/zp/bias
  → FusedMoEKernel / fused_experts 执行 expert GEMM 和 combine
```

如果只记一句话：

```text
MoE 量化的核心，是让 per-expert 的低 bit w13/w2 权重、scale/zero point、routing top-k、expert_map 和 fused MoE kernel 的运行时布局完全对齐。
```

再压缩成最小心智模型：

```text
Router 负责选 expert；
RoutedExperts 负责装 expert 权重；
FusedMoEMethodBase 负责量化参数和 kernel；
FusedMoEQuantConfig 负责告诉 kernel 怎么解释量化；
process_weights_after_loading 负责把 checkpoint 格式变成 kernel 格式。
```
