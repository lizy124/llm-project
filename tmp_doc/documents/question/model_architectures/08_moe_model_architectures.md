# 08. MoE 模型架构如何组织？

源码位置：

- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\layer.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\runner\moe_runner.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\router\fused_moe_router.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\routed_experts.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\fused_moe.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\config.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\mixtral.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\qwen2_moe.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\deepseek_v2.py`

本问题关注：MoE 模型中 router、routed experts、shared experts、fused MoE kernel、expert parallel、EPLB 和权重加载如何接入 vLLM 的模型架构；并对比 Mixtral、Qwen2-MoE、DeepSeek-V2/V3 等典型模型的组织差异。

---

## 1. 一句话回答

MoE 模型把普通 decoder layer 里的 dense MLP 替换成“路由器 + 多个专家 + 输出合并”的稀疏 FFN，vLLM 里这部分主要被收敛到 `FusedMoE(...)` 工厂函数创建的 `MoERunner` 执行管线中。

主链路可以记为：

```text
DecoderLayer
  → Attention
  → post_attention_layernorm
  → MoE block
      → gate / router logits
      → FusedMoE(...) 返回 MoERunner
      → Router.select_experts() 选 top-k experts
      → RoutedExperts.forward_*() 执行专家 MLP
      → fused_experts / MoE kernel
      → shared experts 输出相加
      → TP / EP / SP 归约
  → hidden states
```

所以：

```text
模型文件负责“这一层是不是 MoE、gate 怎么建、shared expert 有没有”；
FusedMoE 工厂负责“构造 router、expert 权重容器、runner”；
MoERunner 负责“路由、dispatch/combine、shared expert、归约”；
fused_moe.py 负责“底层 fused kernel 的两段 expert MLP 计算”。
```

---

## 2. 最小心智模型

普通 MLP：

```text
每个 token 都走同一套 FFN：

hidden_states
  → gate_up_proj
  → activation
  → down_proj
  → output
```

MoE MLP：

```text
每个 token 先被 router 分配到 top-k experts：

hidden_states
  → gate/router logits
  → topk_weights + topk_ids
  → 按 expert 分组执行 expert MLP
  → 按 topk_weights 加权合并
  → 可选 shared experts 相加
  → output
```

在 vLLM 中，大多数 MoE 模型不会在模型文件里手写 token-expert 分组和 kernel 调度，而是调用统一的 `FusedMoE(...)`。

---

## 3. MoE 在 decoder layer 的位置

MoE 通常替代 transformer block 中 attention 后面的 MLP。

以 Mixtral 为例，`MixtralDecoderLayer.forward()` 的结构是：

```text
input_layernorm
  → self_attn
  → post_attention_layernorm
  → block_sparse_moe
```

位置：`vllm/vllm/model_executor/models/mixtral.py:238` 到 `mixtral.py:293`

核心代码形态：

```python
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.block_sparse_moe(hidden_states)
```

位置：`mixtral.py:290` 到 `mixtral.py:292`

Qwen2-MoE 和 DeepSeek 也是同样的 block 位置：

```text
attention 输出
  → post_attention_layernorm
  → self.mlp(hidden_states)
```

区别在于：

```text
Mixtral：几乎每层都是 block_sparse_moe；
Qwen2-MoE：按 decoder_sparse_step / mlp_only_layers 决定 dense 还是 MoE；
DeepSeek：按 first_k_dense_replace / moe_layer_freq 决定 dense 还是 MoE。
```

---

## 4. 通用 MoE block 的输入输出

从模型层看，MoE block 的输入输出非常简单：

```text
输入：hidden_states
输出：新的 hidden_states
```

例如 Mixtral：

```python
router_logits, _ = self.gate(hidden_states)
final_hidden_states = self.experts(hidden_states, router_logits)
```

位置：`mixtral.py:147` 到 `mixtral.py:154`

Qwen2-MoE：

```python
router_logits, _ = self.gate(hidden_states)
final_hidden_states = self.experts(
    hidden_states=hidden_states, router_logits=router_logits
)
```

位置：`qwen2_moe.py:181` 到 `qwen2_moe.py:193`

DeepSeek：

```python
router_logits, _ = self.gate(hidden_states)
final_hidden_states = self.experts(
    hidden_states=hidden_states, router_logits=router_logits
)
```

位置：`deepseek_v2.py:361` 到 `deepseek_v2.py:388`

这说明模型文件只把 hidden states 和 router logits 交给 MoE 执行层，真正的 top-k、dispatch、kernel、combine 都在 fused MoE 层内部完成。

---

## 5. FusedMoE 在新版 vLLM 中是什么

需要注意：当前源码里的 `FusedMoE` 不是一个传统 `class FusedMoE(nn.Module)`，而是一个工厂函数。

定义位置：`vllm/vllm/model_executor/layers/fused_moe/layer.py:102`

它的职责是创建完整的 MoE 执行管线：

```text
FusedMoE(...)
  → make_parallel_config(...)
  → determine_expert_counts(...)
  → ExpertMapManager(...)
  → create_fused_moe_router(...)
  → FusedMoEConfig(...)
  → RoutedExperts(...)
  → MoERunner(...)
  → return runner
```

位置：`layer.py:102` 到 `layer.py:391`

源码注释也明确说它会创建：

```text
- Router：负责 token-to-expert assignment；
- RoutedExperts：持有 expert 权重参数；
- MoERunner：编排完整 forward pass。
```

位置：`layer.py:148` 到 `layer.py:160`

所以模型文件里的：

```python
self.experts = FusedMoE(...)
```

实际含义是：

```text
self.experts 是一个 MoERunner；
它内部包含 router、routed_experts、shared_experts、parallel config 和 quant method。
```

---

## 6. FusedMoE 工厂阶段做了什么

### 6.1 计算 MoE parallel config

入口：`make_parallel_config()`

位置：`layer.py:43` 到 `layer.py:69`

它会根据：

```text
tp_size
dp_size
pcp_size
is_sequence_parallel
parallel_config.enable_expert_parallel
```

生成 `FusedMoEParallelConfig`。

`FusedMoEParallelConfig.make()` 的核心判断是：

```python
use_ep = (
    dp_size_ * pcp_size_ * tp_size_ > 1
    and vllm_parallel_config.enable_expert_parallel
)
```

位置：`config.py:1188` 到 `config.py:1191`

如果不启用 EP：

```text
TP 会被 flatten 到 DP / PCP 维度上，用来切分 expert 权重；
EP size = 1。
```

位置：`config.py:1201` 到 `config.py:1215`

如果启用 EP：

```text
每个 device 拥有一部分完整 experts；
TP size 被置为 1；
EP size 使用 flatten 后的 TP size。
```

位置：`config.py:1216` 到 `config.py:1235`

### 6.2 计算 logical / physical experts

入口：`determine_expert_counts()`

位置：`layer.py:72` 到 `layer.py:99`

它区分：

```text
logical experts：模型配置里的专家数；
physical experts：logical experts + redundant experts；
fused shared experts：ROCm AITER fusion shared experts 特殊路径。
```

普通情况下：

```text
global_num_experts = num_experts + num_redundant_experts
logical_num_experts = num_experts
```

位置：`layer.py:78` 到 `layer.py:79`

### 6.3 创建 ExpertMapManager

`ExpertMapManager` 由 `FusedMoE` 创建：

```python
expert_map_manager = ExpertMapManager(...)
```

位置：`layer.py:250` 到 `layer.py:266`

它负责：

```text
- 当前 rank 拥有哪些 local experts；
- global expert id 到 local expert id 的映射；
- EP / EPLB / round-robin placement 需要的 routing tables；
- redundant experts 场景下 expert_map / expert_mask。
```

后续 `RoutedExperts` 会把这些映射注册成 buffer。

位置：`routed_experts.py:225` 到 `routed_experts.py:240`

### 6.4 创建 Router

如果调用者没有传入 router，`FusedMoE` 会调用：

```python
router = create_fused_moe_router(...)
```

位置：`layer.py:268` 到 `layer.py:294`

传入的关键参数包括：

```text
top_k
global_num_experts
renormalize
use_grouped_topk
num_expert_group / topk_group
scoring_func
routed_scaling_factor
e_score_correction_bias
zero_expert_type
num_logical_experts
```

这说明不同模型的 routing 差异，大多通过 `FusedMoE(...)` 参数进入 router，而不是在 decoder layer 里分散实现。

### 6.5 创建 RoutedExperts

`RoutedExperts` 是 expert 权重容器。

位置：`layer.py:334` 到 `layer.py:366`

它保存：

```text
w13_weight：gate/up 合并后的第一段 expert 权重；
w2_weight：down projection 权重；
scale / zero point / bias 等量化参数；
expert_map / expert_mask；
quant_method。
```

`RoutedExperts` 的类定义位置：`routed_experts.py:43`

源码注释说明它负责：

```text
- loading checkpoint weights into parameters；
- executing routed experts via quant_method.apply()。
```

位置：`routed_experts.py:43` 到 `routed_experts.py:52`

### 6.6 创建 MoERunner

最后，`FusedMoE` 创建并返回：

```python
runner = runner_cls(
    layer_name=layer_name,
    moe_config=moe_config,
    router=router,
    routed_experts=routed_experts,
    ...
)
return runner
```

位置：`layer.py:368` 到 `layer.py:391`

从这一步之后，模型层调用 `self.experts(hidden_states, router_logits)`，实际就是调用 `MoERunner.forward()`。

---

## 7. MoERunner 的 forward 主链路

`MoERunner` 是 MoE 执行的核心编排类。

定义位置：`runner/moe_runner.py:221`

源码注释列出的职责包括：

```text
- expert routing and token dispatching；
- shared experts computation；
- tensor parallel and expert parallel operations；
- multiple quantization methods and optimized kernel selection；
- monolithic and decomposed expert execution paths；
- TP / EP / DP integration。
```

位置：`moe_runner.py:221` 到 `moe_runner.py:239`

### 7.1 forward 的入口

入口：

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
```

位置：`moe_runner.py:628` 到 `moe_runner.py:633`

调用序列：

```text
MoERunner.forward()
  → apply_routed_input_transform()
  → _maybe_pad_hidden_states()
  → self._forward_entry(...)
      → torch.ops.vllm.moe_forward / moe_forward_shared
      → _forward_impl()
  → unpack shared_output / fused_output
  → shared_output + fused_output
  → all-reduce / truncate / zero expert
  → return hidden_states
```

位置：`moe_runner.py:654` 到 `moe_runner.py:716`

### 7.2 custom op 的作用

`MoERunner.forward()` 不直接调用 `_forward_impl()`，而是调用 `_forward_entry`。

位置：`moe_runner.py:290` 到 `moe_runner.py:306`

CUDA 等平台上通常会走：

```text
torch.ops.vllm.moe_forward
或
torch.ops.vllm.moe_forward_shared
```

位置：`moe_runner.py:302` 到 `moe_runner.py:306`

这些 custom op 的实际实现只是从 forward context 找回 layer，然后调用 `_forward_impl()`。

位置：`moe_runner.py:113` 到 `moe_runner.py:166`

这样做的目的包括：

```text
- 让 MoE forward 以 opaque custom op 形式进入 compile / graph；
- 支持带 shared experts 和不带 shared experts 的不同签名；
- 通过 layer_name 找回静态注册的 MoERunner。
```

### 7.3 _forward_impl 的核心步骤

入口：`moe_runner.py:779`

核心流程：

```text
_ensure_moe_quant_config_init()
  → sync shared expert stream
  → 如果 runner 持有 gate，则计算 router_logits
  → _maybe_dispatch(hidden_states, router_logits)
  → _apply_quant_method(...)
  → _maybe_combine(shared_output, hidden_states)
```

位置：`moe_runner.py:779` 到 `moe_runner.py:833`

其中 `_apply_quant_method()` 是路由和专家计算的关键。

---

## 8. Router 如何选择 experts

Router 抽象类是 `FusedMoERouter`。

位置：`router/fused_moe_router.py:12`

它对外暴露：

```python
def select_experts(...) -> tuple[torch.Tensor, torch.Tensor]
```

位置：`fused_moe_router.py:45` 到 `fused_moe_router.py:81`

返回的是：

```text
topk_weights：每个 token 对应 top-k expert 的权重；
topk_ids：每个 token 对应 top-k expert 的 id。
```

位置：`fused_moe_router.py:53` 到 `fused_moe_router.py:60`

兼容性细节：

```text
EPLB 未启用时，返回的 ids 等价于 global logical expert ids；
EPLB 启用时，ids 可能经过 physical/redundant expert 映射。
```

位置：`fused_moe_router.py:62` 到 `fused_moe_router.py:64`

在 `MoERunner._apply_quant_method()` 中，如果不是 monolithic kernel，会显式调用：

```python
topk_weights, topk_ids = self.router.select_experts(...)
```

位置：`moe_runner.py:559` 到 `moe_runner.py:565`

然后把 `topk_weights/topk_ids` 交给 `RoutedExperts.forward_modular()`。

位置：`moe_runner.py:567` 到 `moe_runner.py:573`

---

## 9. RoutedExperts 如何持有和执行专家权重

`RoutedExperts` 是 routed expert 权重和执行逻辑的容器。

定义位置：`routed_experts.py:43`

它初始化时会：

```text
1. 保存 moe_config / quant_config / expert_map_manager；
2. 注册 expert_map / expert_mask / routing tables；
3. 选择 quant_method；
4. 根据 quant_method 创建 expert 权重；
5. 提供 weight_loader 给 checkpoint 加载使用。
```

位置：`routed_experts.py:54` 到 `routed_experts.py:169`

关键字段包括：

```text
global_num_experts：全局物理专家数；
local_num_experts：当前 rank 持有的专家数；
expert_map：global expert id 到本地 expert slot 的映射；
quant_method：实际执行 expert 计算的量化/非量化方法。
```

位置：`routed_experts.py:81` 到 `routed_experts.py:118`

`expert_map` 属性会根据 ROCm AITER 路径选择 `_expert_map` 或 `expert_mask`。

位置：`routed_experts.py:219` 到 `routed_experts.py:223`

这解释了为什么模型加载 expert 权重时，不是简单按 checkpoint 名字直接塞张量，而是要通过 expert-aware `weight_loader`。

---

## 10. fused experts kernel 做了什么

底层通用函数在 `fused_moe.py`。

外层入口：

```python
def fused_experts(...)
```

位置：`fused_moe.py:1474` 到 `fused_moe.py:1515`

它把参数转给 custom op：

```python
torch.ops.vllm.fused_experts(...)
```

位置：`fused_moe.py:1490` 到 `fused_moe.py:1515`

真正实现：

```python
def fused_experts_impl(...)
```

位置：`fused_moe.py:1537` 到 `fused_moe.py:1740`

### 10.1 fused_experts_impl 的主流程

可以压缩成：

```text
hidden_states + topk_ids/topk_weights
  → moe_kernel_quantize_input(hidden_states)
  → _prepare_expert_assignment(topk_ids)
  → dispatch_fused_moe_kernel(w1)
  → apply_moe_activation()
  → moe_kernel_quantize_input(intermediate)
  → dispatch_fused_moe_kernel(w2)
  → ops.moe_sum()
  → out_hidden_states
```

对应位置：

```text
输入量化：fused_moe.py:1651 到 fused_moe.py:1657
expert assignment：fused_moe.py:1659 到 fused_moe.py:1670
第一段 w1 kernel：fused_moe.py:1672 到 fused_moe.py:1694
激活：fused_moe.py:1696 到 fused_moe.py:1698
第二段输入量化：fused_moe.py:1700 到 fused_moe.py:1706
第二段 w2 kernel：fused_moe.py:1711 到 fused_moe.py:1733
top-k 汇总：fused_moe.py:1735 到 fused_moe.py:1738
```

### 10.2 为什么是两段 expert MLP

MoE expert 本质仍是 FFN：

```text
w1 / gate_up projection
  → activation
  → w2 / down projection
```

所以 fused kernel 也分两次 matmul：

```text
第一次：hidden_states → intermediate_cache1
第二次：intermediate_cache2 → intermediate_cache3
```

最后 `ops.moe_sum()` 按 top-k 对每个 token 的 expert 输出求和。

### 10.3 expert assignment 如何准备

入口：`_prepare_expert_assignment()`

位置：`fused_moe.py:1425` 到 `fused_moe.py:1471`

它有两类路径：

```text
naive_block_assignment：小 batch / 稀疏激活场景下跳过 moe_align_block_size；
moe_align_block_size：按 expert 对 token 排序、padding、对齐 block。
```

位置：`fused_moe.py:1443` 到 `fused_moe.py:1471`

这一步的输出会喂给 Triton / CUDA kernel：

```text
sorted_token_ids
expert_ids
num_tokens_post_padded
```

---

## 11. shared experts 如何接入

shared experts 是所有 token 都会走的一组 dense/shared FFN，通常和 routed expert 输出相加。

通用结构：

```text
routed expert output：按 top-k sparse routing 计算；
shared expert output：所有 token 都计算；
最终输出：shared_output + fused_output。
```

在 `MoERunner.forward()` 尾部：

```python
if shared_output is not None:
    result = shared_output + fused_output
else:
    result = fused_output
```

位置：`moe_runner.py:709` 到 `moe_runner.py:712`

### 11.1 Qwen2-MoE 的 shared experts

`Qwen2MoeSparseMoeBlock` 总是创建 `shared_expert_gate`：

```python
self.shared_expert_gate = ReplicatedLinear(...)
```

位置：`qwen2_moe.py:149` 到 `qwen2_moe.py:155`

如果 `shared_expert_intermediate_size > 0`，再创建：

```python
self.shared_expert = Qwen2MoeMLP(...)
```

位置：`qwen2_moe.py:157` 到 `qwen2_moe.py:168`

然后传入：

```python
self.experts = FusedMoE(shared_experts=self.shared_expert, ...)
```

位置：`qwen2_moe.py:170` 到 `qwen2_moe.py:179`

这意味着 Qwen2-MoE 的 shared expert 由 `MoERunner` 统一编排，而不是在 sparse block 里手动相加。

### 11.2 DeepSeek 的 shared experts

`DeepseekV2MoE` 中：

```python
if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:
    self.shared_experts = None
else:
    self.shared_experts = DeepseekV2MLP(...)
```

位置：`deepseek_v2.py:311` 到 `deepseek_v2.py:325`

然后同样传给 `FusedMoE`：

```python
self.experts = FusedMoE(
    shared_experts=self.shared_experts,
    ...
)
```

位置：`deepseek_v2.py:326` 到 `deepseek_v2.py:351`

特殊点是 ROCm AITER fusion shared experts 启用时，shared experts 可能不作为普通 `DeepseekV2MLP` 存在，而是融合进 routed expert 路径。

---

## 12. Expert Parallel 和 EPLB 如何接入

MoE 的并行和普通 dense MLP 最大区别是：expert 可以按专家维度切分。

### 12.1 TP 模式和 EP 模式的区别

不启用 EP 时：

```text
每个 expert 的权重像普通 TP 一样切到多个 rank；
每个 rank 都参与同一批 experts 的计算；
最后需要 tensor parallel all-reduce。
```

启用 EP 时：

```text
每个 rank 持有一部分完整 experts；
token 需要被 dispatch 到拥有对应 expert 的 rank；
执行后再 combine 回来。
```

`FusedMoEParallelConfig.make()` 中明确写道：

```text
In EP, each device owns a set of experts fully.
There is no tensor parallel.
```

位置：`config.py:1216` 到 `config.py:1220`

### 12.2 MoERunner 中的 dispatch / combine

MoERunner 的 `_maybe_dispatch()` 会处理 naive DP/EP dispatch：

```python
get_ep_group().dispatch_router_logits(...)
```

位置：`moe_runner.py:724` 到 `moe_runner.py:741`

`_maybe_combine()` 会处理 EP combine：

```python
hidden_states = get_ep_group().combine(...)
```

位置：`moe_runner.py:757` 到 `moe_runner.py:765`

PCP 场景还会使用 all-gather / reduce-scatter。

位置：`moe_runner.py:745` 到 `moe_runner.py:771`

### 12.3 EPLB 和 redundant experts

EPLB 是 Expert Parallelism Load Balancer。

在 `FusedMoE(...)` 中：

```python
if enable_eplb:
    eplb_state = EplbLayerState()
else:
    assert num_redundant_experts == 0
```

位置：`layer.py:233` 到 `layer.py:248`

这说明：

```text
redundant experts 只有在 EPLB 启用时才允许存在。
```

Mixtral 和 DeepSeek 都会从 parallel config 读取：

```text
parallel_config.eplb_config.num_redundant_experts
```

Mixtral 位置：`mixtral.py:106` 到 `mixtral.py:119`

DeepSeek 位置：`deepseek_v2.py:286` 到 `deepseek_v2.py:299`

当物理专家数变化时，模型的 `update_physical_experts_metadata()` 会更新各层 MoE 元数据，并调用：

```python
moe.experts.update_expert_map()
```

Mixtral 位置：`mixtral.py:542` 到 `mixtral.py:560`

DeepSeek 位置：`deepseek_v2.py:1606` 到 `deepseek_v2.py:1620`

---

## 13. quantized MoE 如何接入

量化 MoE 不是模型文件单独处理，而是通过 `quant_config` 进入 `FusedMoE` / `RoutedExperts` / `quant_method`。

`RoutedExperts` 初始化时：

```python
quant_method = quant_config.get_quant_method(self, prefix)
```

如果没有量化方法，则使用：

```python
UnquantizedFusedMoEMethod
```

位置：`routed_experts.py:180` 到 `routed_experts.py:196`

底层 `fused_experts()` 支持多种量化标志：

```text
use_fp8_w8a8
use_int8_w8a8
use_int8_w8a16
use_int4_w4a16
per_channel_quant
block_shape
w1_scale / w2_scale
w1_zp / w2_zp
a1_scale / a2_scale
```

位置：`fused_moe.py:1474` 到 `fused_moe.py:1515`

在 `fused_experts_impl()` 中，会先对输入量化，再调用对应 kernel：

```text
moe_kernel_quantize_input()
  → dispatch_fused_moe_kernel()
```

位置：`fused_moe.py:1651` 到 `fused_moe.py:1694`

第二段 `w2` 前也会再次量化中间激活。

位置：`fused_moe.py:1700` 到 `fused_moe.py:1733`

所以可以把量化 MoE 理解为：

```text
模型层不关心具体 FP8 / INT8 / INT4 kernel；
RoutedExperts 通过 quant_method 创建权重和 scale；
fused_experts_impl 根据 quant_config 选择量化输入和 kernel 参数。
```

---

## 14. Mixtral 的 MoE 组织

Mixtral 是最典型、最直观的 routed MoE。

### 14.1 MixtralMoE

定义位置：`mixtral.py:77`

它创建：

```python
self.gate = ReplicatedLinear(...)
self.experts = FusedMoE(...)
```

位置：`mixtral.py:121` 到 `mixtral.py:145`

forward 中：

```text
hidden_states reshape
  → gate(hidden_states) 得到 router_logits
  → experts(hidden_states, router_logits)
  → reshape 回原形状
```

位置：`mixtral.py:147` 到 `mixtral.py:154`

### 14.2 MixtralDecoderLayer

`MixtralDecoderLayer` 里 MoE 字段叫：

```python
self.block_sparse_moe = MixtralMoE(...)
```

位置：`mixtral.py:259` 到 `mixtral.py:267`

这层的 MLP 阶段直接调用：

```python
hidden_states = self.block_sparse_moe(hidden_states)
```

位置：`mixtral.py:290` 到 `mixtral.py:292`

### 14.3 MixtralForCausalLM 的 MoE 元数据

`MixtralForCausalLM` 初始化时会遍历 decoder layers，收集所有 MoE 层：

```python
self.moe_layers.append(layer.block_sparse_moe.experts)
```

位置：`mixtral.py:515` 到 `mixtral.py:529`

并记录：

```text
num_logical_experts
num_physical_experts
num_local_physical_experts
num_routed_experts
num_redundant_experts
num_shared_experts = 0
```

位置：`mixtral.py:534` 到 `mixtral.py:540`

这说明 Mixtral 没有 shared experts，是纯 routed experts 模型。

---

## 15. Qwen2-MoE 的组织差异

Qwen2-MoE 的特点是：不是每层都一定是 MoE，并且可能有 shared expert gate。

### 15.1 SparseMoeBlock

定义位置：`qwen2_moe.py:125`

核心组件：

```text
gate：router logits；
shared_expert_gate：shared expert 的 gate；
shared_expert：可选 dense shared expert；
experts：FusedMoE runner。
```

位置：`qwen2_moe.py:141` 到 `qwen2_moe.py:179`

forward 仍然是：

```text
gate(hidden_states)
  → FusedMoE(hidden_states, router_logits)
```

位置：`qwen2_moe.py:181` 到 `qwen2_moe.py:193`

### 15.2 哪些层是 MoE

`Qwen2MoeDecoderLayer` 根据 layer index 判断：

```python
if (layer_idx not in mlp_only_layers) and (
    config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
):
    self.mlp = Qwen2MoeSparseMoeBlock(...)
else:
    self.mlp = Qwen2MoeMLP(...)
```

位置：`qwen2_moe.py:311` 到 `qwen2_moe.py:330`

所以 Qwen2-MoE 可能是：

```text
部分层 dense MLP；
部分层 sparse MoE；
某些模型配置还显式指定 mlp_only_layers。
```

### 15.3 权重加载细节

Qwen2-MoE 的 `load_weights()` 有一个重要保护：

```text
如果 checkpoint 名字里有 mlp.experts，不能先按 gate_up_proj 规则替换；
否则专家权重名会被重复改写。
```

位置：`qwen2_moe.py:446` 到 `qwen2_moe.py:458`

另一个细节是 GGUF shared expert gate：

```python
if "mlp.shared_expert_gate" in name and len(loaded_weight.shape) == 1:
    loaded_weight = loaded_weight[None, :]
```

位置：`qwen2_moe.py:522` 到 `qwen2_moe.py:527`

---

## 16. DeepSeek-V2/V3 的 MoE 组织差异

DeepSeek 的 MoE 逻辑比 Mixtral / Qwen2-MoE 更复杂，主要体现在 grouped top-k、shared experts、sequence parallel、EPLB、ROCm AITER、MLA/MHA 和特殊权重加载。

### 16.1 DeepseekV2MoE

定义位置：`deepseek_v2.py:246`

它创建：

```text
GateLinear：router；
shared_experts：可选 DeepseekV2MLP；
FusedMoE：routed experts 执行管线。
```

位置：`deepseek_v2.py:274` 到 `deepseek_v2.py:351`

传给 `FusedMoE` 的关键参数包括：

```text
shared_experts
use_grouped_topk=True
num_expert_group
topk_group
scoring_func
routed_scaling_factor
apply_routed_scale_to_output
e_score_correction_bias
enable_eplb
num_redundant_experts
is_sequence_parallel
n_shared_experts
router_logits_dtype
```

位置：`deepseek_v2.py:326` 到 `deepseek_v2.py:351`

这说明 DeepSeek 的 MoE 不是简单 top-k，而是把 grouped top-k、修正 bias、routing scale、SP、EPLB 等都放进统一 runner。

### 16.2 DeepSeek 的 sequence parallel

DeepSeek MoE 支持 sequence parallel：

```python
self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
```

位置：`deepseek_v2.py:266`

forward 中如果启用，会先 chunk：

```python
hidden_states = sequence_parallel_chunk(hidden_states)
```

位置：`deepseek_v2.py:365` 到 `deepseek_v2.py:370`

MoE 输出后再 gather：

```python
final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
final_hidden_states = final_hidden_states[:num_tokens]
```

位置：`deepseek_v2.py:382` 到 `deepseek_v2.py:386`

### 16.3 哪些层是 DeepSeek MoE

`DeepseekV2DecoderLayer` 判断：

```python
if (
    config.n_routed_experts is not None
    and layer_idx >= config.first_k_dense_replace
    and layer_idx % moe_layer_freq == 0
):
    self.mlp = DeepseekV2MoE(...)
else:
    self.mlp = DeepseekV2MLP(...)
```

位置：`deepseek_v2.py:1155` 到 `deepseek_v2.py:1173`

所以 DeepSeek 可能前若干层是 dense，后续按频率插入 MoE。

### 16.4 DeepSeekForCausalLM 的 MoE 元数据

`DeepseekV2ForCausalLM.set_moe_parameters()` 会遍历层：

```text
如果 layer.mlp 是 DeepseekV2MoE：
  加入 moe_mlp_layers；
  加入 moe_layers；
  用 example_moe 提取 num experts 元数据。
```

位置：`deepseek_v2.py:1686` 到 `deepseek_v2.py:1705`

提取的元数据包括：

```text
num_logical_experts
num_physical_experts
num_local_physical_experts
num_routed_experts
num_shared_experts
num_redundant_experts
```

位置：`deepseek_v2.py:1587` 到 `deepseek_v2.py:1605`

---

## 17. MoE 权重加载如何组织

MoE 权重加载比 dense MLP 更复杂，因为 checkpoint 里通常是：

```text
experts.0.gate_proj.weight
experts.0.up_proj.weight
experts.0.down_proj.weight
experts.1.gate_proj.weight
...
```

而 vLLM fused MoE 内部通常是：

```text
w13_weight：gate/up 合并参数；
w2_weight：down 参数；
每个 local expert 有自己的 slot；
可能还有 FP8 scale / activation scale / redundant experts。
```

因此模型类会提供 `get_expert_mapping()`。

### 17.1 通用映射函数

通用函数：

```python
fused_moe_make_expert_params_mapping(...)
```

位置：`layer.py:394` 到 `layer.py:412`

它最终委托给：

```text
RoutedExperts.make_expert_params_mapping(...)
```

位置：`layer.py:403` 到 `layer.py:412`

返回 tuple 形态：

```text
(param_name, weight_name, expert_id, shard_id)
```

模型加载时用它判断：

```text
checkpoint 中某个 expert 权重应该加载到 fused MoE 的哪个参数、哪个 expert、哪个 shard。
```

### 17.2 Mixtral 权重映射

Mixtral checkpoint 使用：

```text
w1：gate projection；
w2：down projection；
w3：up projection。
```

`MixtralModel.get_expert_mapping()`：

```python
ckpt_gate_proj_name="w1"
ckpt_down_proj_name="w2"
ckpt_up_proj_name="w3"
```

位置：`mixtral.py:367` 到 `mixtral.py:377`

加载时遍历 `expert_params_mapping`，调用 expert-aware `weight_loader`。

位置：`mixtral.py:413` 到 `mixtral.py:449`

### 17.3 Qwen2-MoE 权重映射

Qwen2-MoE checkpoint 使用：

```text
gate_proj
down_proj
up_proj
```

位置：`qwen2_moe.py:421` 到 `qwen2_moe.py:430`

加载时同样走 expert mapping。

位置：`qwen2_moe.py:475` 到 `qwen2_moe.py:498`

### 17.4 DeepSeek 权重映射

DeepSeek 的 `load_weights()` 更复杂，因为还处理 MLA/MHA、fused indexer、shared experts、FP8 kv scale 等。

expert mapping 构造位置：`deepseek_v2.py:1386` 到 `deepseek_v2.py:1400`

普通 expert 加载位置：`deepseek_v2.py:1511` 到 `deepseek_v2.py:1556`

ROCm AITER fusion shared experts 特殊处理：

```text
checkpoint 中的 mlp.shared_experts.* 可能是一个 widened tensor；
加载时按 n_shared_experts 切成多个 chunk；
然后伪装成 mlp.experts.{n_routed_experts + j}.*；
再走 expert mapping 加载。
```

位置：`deepseek_v2.py:1466` 到 `deepseek_v2.py:1510`

---

## 18. MoE 和 Pipeline Parallel 的关系

MoE 本身位于模型层内部，所以它和普通 MLP 一样遵守 PP 分层逻辑。

以 MixtralModel 为例：

```text
first PP rank：从 input_ids / inputs_embeds 得到 hidden_states；
中间 PP rank：从 IntermediateTensors 取 hidden_states / residual；
遍历当前 rank 持有的 layers；
非 last PP rank：返回 IntermediateTensors；
last PP rank：norm 后返回 hidden_states。
```

位置：`mixtral.py:341` 到 `mixtral.py:365`

Qwen2-MoE 同样：`qwen2_moe.py:395` 到 `qwen2_moe.py:419`

DeepSeek 同样：`deepseek_v2.py:1297` 到 `deepseek_v2.py:1353`

这说明：

```text
MoE 不改变 PP 的基本协议；
它只改变某些 decoder layer 内部的 MLP 实现；
非最后 PP rank 仍然只返回 IntermediateTensors，不计算 logits。
```

---

## 19. MoE 和 ModelRunner 的关系

从执行层看，MoE 并不是一个单独的调度阶段。

主链路仍然是：

```text
GPUModelRunner.execute_model()
  → _prepare_inputs()
  → _build_attention_metadata()
  → set_forward_context(...)
  → _model_forward()
  → model.forward()
  → decoder layers
  → MoE layer inside decoder
  → hidden_states
  → compute_logits / pool
```

MoE 发生在模型 forward 内部。

因此：

```text
Scheduler 不直接感知某个 token 被路由到了哪个 expert；
ModelRunner 只调用模型 forward；
MoE routing / dispatch / kernel 都封装在模型层里的 MoERunner。
```

但 vLLM 的输出对象预留了 `routed_experts` 字段，用于部分场景记录 routed expert 信息。

---

## 20. 三类典型模型对比

### 20.1 Mixtral

```text
特点：最标准 routed MoE。

MoE block：MixtralMoE
router：ReplicatedLinear
experts：FusedMoE
shared experts：无
routing：top-k experts per token
权重名：w1 / w2 / w3
```

关键位置：

```text
MixtralMoE：mixtral.py:77 到 mixtral.py:154
MixtralDecoderLayer：mixtral.py:238 到 mixtral.py:293
MoE 元数据：mixtral.py:515 到 mixtral.py:540
权重映射：mixtral.py:367 到 mixtral.py:377
```

### 20.2 Qwen2-MoE

```text
特点：dense 层和 sparse MoE 层混合，支持 shared expert。

MoE block：Qwen2MoeSparseMoeBlock
router：ReplicatedLinear gate
shared expert gate：ReplicatedLinear
shared expert：Qwen2MoeMLP，可选
experts：FusedMoE(shared_experts=...)
层选择：decoder_sparse_step / mlp_only_layers
权重名：gate_proj / up_proj / down_proj
```

关键位置：

```text
SparseMoeBlock：qwen2_moe.py:125 到 qwen2_moe.py:193
层选择：qwen2_moe.py:311 到 qwen2_moe.py:330
shared expert：qwen2_moe.py:149 到 qwen2_moe.py:179
权重加载：qwen2_moe.py:432 到 qwen2_moe.py:534
```

### 20.3 DeepSeek-V2/V3

```text
特点：最复杂，支持 grouped top-k、shared experts、SP、EPLB、ROCm AITER、MLA/MHA 差异。

MoE block：DeepseekV2MoE
router：GateLinear
experts：FusedMoE(gate=..., use_grouped_topk=True, ...)
shared experts：DeepseekV2MLP 或 fused shared experts
层选择：first_k_dense_replace / moe_layer_freq
权重名：gate_proj / up_proj / down_proj
特殊加载：shared experts 可切 chunk 后当作追加 expert 加载
```

关键位置：

```text
DeepseekV2MoE：deepseek_v2.py:246 到 deepseek_v2.py:388
层选择：deepseek_v2.py:1155 到 deepseek_v2.py:1173
SP：deepseek_v2.py:365 到 deepseek_v2.py:386
MoE 元数据：deepseek_v2.py:1587 到 deepseek_v2.py:1705
权重加载：deepseek_v2.py:1355 到 deepseek_v2.py:1578
```

---

## 21. 容易疑惑的点

### 21.1 FusedMoE 是不是一个普通 layer 类？

当前源码里不是。

它是工厂函数，返回 `MoERunner`。

位置：`layer.py:102` 到 `layer.py:147`

### 21.2 router logits 是在哪里算的？

有两种形态：

```text
模型 block 外部先算：Mixtral / Qwen2-MoE 常见；
MoERunner 内部持有 gate 再算：DeepSeek 传 gate 给 FusedMoE 后可走 internal router。
```

DeepSeek forward 中会判断：

```python
if self.experts.is_internal_router:
    final_hidden_states = self.experts(
        hidden_states=hidden_states, router_logits=hidden_states
    )
else:
    router_logits, _ = self.gate(hidden_states)
```

位置：`deepseek_v2.py:372` 到 `deepseek_v2.py:380`

`MoERunner._forward_impl()` 里如果 `self.gate is not None`，会实际计算 gate。

位置：`moe_runner.py:804` 到 `moe_runner.py:813`

### 21.3 expert weights 是每个 expert 一个 Linear 吗？

逻辑上可以这么理解，但实现上不是 Python list 里很多 `Linear`。

vLLM 把 experts 的权重组织成 fused 参数，由 `RoutedExperts` 持有，并通过 `quant_method` 创建参数。

位置：`routed_experts.py:150` 到 `routed_experts.py:169`

### 21.4 shared expert 和 routed expert 谁先算？

从语义上看最终是相加：

```text
output = shared_output + fused_output
```

实际执行时，`MoERunner` 支持 overlap：shared experts 可以在独立 stream 上和 routed experts 并行。

相关位置：

```text
SharedExperts 包装：moe_runner.py:277 到 moe_runner.py:285
触发 shared experts：moe_runner.py:525 到 moe_runner.py:532
overlap 调度：moe_runner.py:547 到 moe_runner.py:578
输出相加：moe_runner.py:709 到 moe_runner.py:712
```

### 21.5 EP 和 TP 的区别是什么？

```text
TP：每个 expert 的矩阵被切分到多个 rank；
EP：不同 rank 持有不同 experts，token 需要按 expert dispatch/combine。
```

`FusedMoEParallelConfig.make()` 中 EP 会把 TP size 置为 1，把 flatten 后的设备数作为 EP size。

位置：`config.py:1216` 到 `config.py:1235`

### 21.6 为什么 MoE 权重加载这么复杂？

因为 checkpoint 的 expert 权重是按 expert id 命名的，而 vLLM 内部会做：

```text
fused gate/up 权重；
local expert sharding；
EP/EPLB expert_map；
量化 scale / zero point；
shared expert fusion；
PP missing layer 跳过。
```

所以模型必须提供 `get_expert_mapping()`，再由 expert-aware loader 精确加载。

---

## 22. 总结

MoE 模型架构可以压缩成：

```text
DecoderLayer
  → Attention
  → MoE block
      → gate / router logits
      → FusedMoE 工厂创建 MoERunner
      → Router.select_experts()
      → RoutedExperts 持有和加载 expert 权重
      → fused_experts_impl 两段 expert MLP kernel
      → shared experts 相加
      → TP / EP / SP 归约
  → hidden states
```

如果只记一条主线：

```text
vLLM 把各模型不同的 MoE 写法统一收敛到 FusedMoE / MoERunner：模型层只决定“何时用 MoE、gate/shared experts 怎么配置”，通用 fused MoE 层负责“选专家、跑专家、合并输出、处理并行和量化”。
```

再压缩成最小心智模型：

```text
router 产生 top-k；
RoutedExperts 管 expert 权重；
MoERunner 编排 dispatch / shared experts / reduce；
fused_moe kernel 完成高性能 expert MLP；
模型文件负责把这些组件嵌回 decoder layer。
```
