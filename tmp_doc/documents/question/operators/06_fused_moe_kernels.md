# 06. MoE fused 算子如何执行 expert routing？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\fused_moe\layer.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\fused_moe\runner\moe_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\fused_moe\routed_experts.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\fused_moe\router\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\fused_moe\fused_moe.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\fused_moe\modular_kernel.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\fused_moe\experts\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\fused_moe\prepare_finalize\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\quantization\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\_custom_ops.py`
- `D:\lzy\project\kv_pool\code\vllm\csrc\moe\`

本问题关注：MoE 的 router logits、top-k、token dispatch、expert 权重、grouped GEMM、activation、expert combine、量化、expert parallel 等环节，如何从模型 layer 进入 `FusedMoE`，再由 router、`RoutedExperts`、`MoERunner` 和底层 fused kernel 协作完成。

---

## 1. 一句话回答

vLLM 里的 Fused MoE 不是一个单独的大函数，而是一条由三类对象拼起来的执行管线：

```text
FusedMoE factory
  → FusedMoERouter：从 router_logits 选 top-k expert
  → RoutedExperts：持有 expert 权重和 quant_method
  → MoERunner：组织 dispatch / expert compute / combine / all-reduce
  → fused MoE kernel：按 expert 分组执行 w13、activation、w2，再把 top-k 输出加权合并
```

最小 forward 链路是：

```text
hidden_states + router_logits
  → router.select_experts()
      → topk_weights / topk_ids
  → routed_experts.forward_modular()
      → quant_method.apply()
      → moe_kernel.apply() 或 fused_experts()
  → token 按 expert 分组 / padding
  → grouped GEMM: w13
  → fused activation: silu/gelu/swiglu...
  → grouped GEMM: w2
  → moe_sum / combine
  → TP/EP/SP reduce 或 combine
  → output hidden_states
```

如果底层是 monolithic kernel，则 routing 可以被 kernel 内部吞掉：

```text
hidden_states + router_logits
  → routed_experts.forward_monolithic()
  → quant_method.apply_monolithic()
  → kernel 内部完成 routing + expert compute + combine
```

---

## 2. 先给结论：MoE 这几层分别负责什么

### 2.1 `FusedMoE()` 是 factory，不是真正 forward 函数

位置：`code/vllm/vllm/model_executor/layers/fused_moe/layer.py:103`

`FusedMoE()` 会创建三样东西：

- `FusedMoERouter`：负责 routing 语义，例如 softmax top-k、sigmoid top-k、grouped top-k、自定义 routing、EPLB routing。
- `RoutedExperts`：负责 expert 权重参数、权重加载、quant_method 初始化。
- `MoERunner`：负责 forward 管线，把 router、expert compute、shared experts、EP/TP 通信串起来。

所以模型里写的：

```text
self.moe = FusedMoE(...)
```

最后拿到的通常是一个 `MoERunner`，不是一个旧式 `nn.Module` 中只包含一个 kernel call 的对象。

### 2.2 router 只负责选 expert，不负责执行 expert MLP

router 的抽象在：`code/vllm/vllm/model_executor/layers/fused_moe/router/fused_moe_router.py:12`

核心接口是：

```python
select_experts(hidden_states, router_logits, topk_indices_dtype, input_ids=None)
  → topk_weights, topk_ids
```

默认 `FusedTopKRouter` 会调用 fused top-k op：

- `ops.topk_softmax`
- `ops.topk_sigmoid`

对应位置：

- `code/vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:69`
- `code/vllm/vllm/_custom_ops.py:2380`
- `code/vllm/vllm/_custom_ops.py:2398`

### 2.3 `RoutedExperts` 持有 expert 权重和 quant_method

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:43`

它持有的典型权重是：

```text
w13_weight: [num_local_experts, 2 * intermediate, hidden]
w2_weight:  [num_local_experts, hidden, intermediate]
```

其中 `w13` 是 gate/up projection 的 fused 权重，`w2` 是 down projection。

如果有量化，还会多出：

```text
w1_scale / w2_scale
w1_zp / w2_zp
a1_scale / a2_scale
block_shape / group metadata
packed weight / reordered weight
```

### 2.4 `FusedMoEMethodBase` 是 MoE 的量化/后端策略接口

位置：`code/vllm/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py:31`

它继承 `QuantizeMethodBase`，但专门服务 MoE expert 权重和 grouped GEMM。

关键接口：

```text
create_weights()
get_fused_moe_quant_config()
apply()
apply_monolithic()
process_weights_after_loading()
```

未量化路径是：

```text
UnquantizedFusedMoEMethod
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:45`

量化路径则来自 FP8、INT8、W4A16、MXFP4、NVFP4、compressed-tensors 等 quant method。

### 2.5 `MoERunner` 才是 forward 管线的组织者

位置：`code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:221`

它做这些事：

- 可选地执行 gate 得到 `router_logits`
- 可选地执行 routed input/output transform
- 可选地执行 shared experts
- 在 modular 路径下调用 router 得到 `topk_weights / topk_ids`
- 调用 `RoutedExperts.forward_modular()` 或 `forward_monolithic()`
- 处理 DP/EP/PCP/SP 下的 dispatch/combine
- 处理 TP/EP all-reduce
- 处理 zero expert / shared expert 输出合并

---

## 3. 整体流程图

标准 modular 路径可以理解成：

```text
model decoder layer
  → gate(hidden_states) 或外部传入 router_logits
  → MoERunner.forward(hidden_states, router_logits)
      → torch.ops.vllm.moe_forward(...)
      → MoERunner._forward_impl(...)
          → _maybe_dispatch()
              → DP/EP/PCP/SP token 通信
          → _apply_quant_method()
              → router.select_experts()
                  → topk_weights, topk_ids
              → routed_experts.forward_modular()
                  → quant_method.apply()
                      → moe_kernel.apply()
                      → prepare_finalize + experts GEMM + finalize
          → _maybe_combine()
      → shared_output + fused_output
      → _maybe_reduce_final_output()
  → next layer
```

旧的或特殊后端 monolithic 路径是：

```text
MoERunner._apply_quant_method()
  → routed_experts.forward_monolithic(x, router_logits)
      → quant_method.apply_monolithic()
          → kernel 内部完成 top-k / dispatch / expert compute / combine
```

---

## 4. `FusedMoE()` 初始化时做了什么

位置：`code/vllm/vllm/model_executor/layers/fused_moe/layer.py:103`

### 4.1 先确定并行配置

`make_parallel_config()` 会根据：

- TP size
- DP size
- PCP size
- sequence parallel
- vLLM 全局 `ParallelConfig`

构造 `FusedMoEParallelConfig`。

位置：`code/vllm/vllm/model_executor/layers/fused_moe/layer.py:43`

这个配置会决定后续：

```text
是否使用 EP
local_num_experts 是多少
是否需要 dispatch/combine
是否需要 sequence parallel 上下文
```

### 4.2 再确定 expert 数量

`determine_expert_counts()` 会区分：

```text
global_num_experts：包含 redundant experts 后的物理 expert 总数
logical_num_experts：模型语义上的 expert 数
num_fused_shared_experts：ROCm aiter shared expert fusion 相关数量
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/layer.py:72`

### 4.3 `ExpertMapManager` 负责 expert placement

位置：`code/vllm/vllm/model_executor/layers/fused_moe/layer.py:255`

它会生成：

- `expert_map`
- `expert_mask`
- routing tables
- local/global expert 映射

这些信息后面会被 `RoutedExperts` 注册成 buffer，并传给 kernel。

### 4.4 创建 router

如果调用方没有传入 router，factory 会调用：

```python
create_fused_moe_router(...)
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/layer.py:271`

router 会根据参数选择不同 routing 方式：

- 普通 fused top-k
- grouped top-k
- top-k with bias
- custom routing
- routing simulator
- zero expert router
- EPLB aware routing

### 4.5 创建 `FusedMoEConfig`

位置：`code/vllm/vllm/model_executor/layers/fused_moe/layer.py:309`

它把 MoE kernel 需要的静态信息集中起来：

```text
num_experts
experts_per_token
hidden_dim
intermediate_size
num_local_experts
parallel config
input dtype
moe_backend
routing_method
activation
max_num_tokens
CUDA graph capture size
```

### 4.6 创建 `RoutedExperts` 和 `MoERunner`

`RoutedExperts` 在创建时会立即：

1. 选择 quant_method
2. 根据 quant_method 可能 round up hidden/intermediate size
3. 调用 `quant_method.create_weights()` 注册 expert 参数

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:114` 到 `code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:168`

最后 `MoERunner` 把 router 和 routed experts 绑定起来。

---

## 5. forward 入口为什么还有一层 `torch.ops.vllm.moe_forward`

`MoERunner.forward()` 不是直接调用 `_forward_impl()`，而是调用：

```text
self._forward_entry
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:628`

在 CUDA 类平台上，`_forward_entry` 通常是：

```text
torch.ops.vllm.moe_forward
torch.ops.vllm.moe_forward_shared
```

注册位置：`code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:193`

这层 custom op 的目的不是做计算，而是：

```text
让 torch.compile / CUDA graph / MoE-LoRA dual-stream 路径把 MoE forward 当成 opaque op，
再通过 forward context 里的 layer_name 找回真实 MoERunner 实例。
```

实际执行仍然会回到：

```text
MoERunner._forward_impl()
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:779`

---

## 6. router 如何产生 top-k expert

### 6.1 router 输入输出

输入：

```text
hidden_states: [num_tokens, hidden]
router_logits: [num_tokens, num_experts]
```

输出：

```text
topk_weights: [num_tokens, top_k]
topk_ids:     [num_tokens, top_k]
```

`topk_ids` 是每个 token 选择的 expert id；`topk_weights` 是对应 expert 输出的合并权重。

### 6.2 默认 fused top-k

位置：`code/vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:69`

默认路径会分配三个张量：

```text
topk_weights
topk_ids
token_expert_indices
```

然后根据 scoring function 调用：

```text
ops.topk_softmax(...)
ops.topk_sigmoid(...)
```

这一步把 softmax/sigmoid、top-k 选择、可选 renormalize 放在一个 fused op 里，避免在 Python/Torch 里拆成多个临时张量和 kernel launch。

### 6.3 grouped top-k 和 bias top-k

一些模型不是从所有 experts 里直接 top-k，而是先按 group 过滤或加 correction bias。

对应 router 文件包括：

```text
router/grouped_topk_router.py
router/fused_topk_bias_router.py
router/custom_routing_router.py
```

这些变体仍然产出同一组接口：

```text
topk_weights, topk_ids
```

这样后面的 `RoutedExperts` 不需要关心 routing 细节。

### 6.4 EPLB 下的 expert id

启用 EPLB 或 redundant experts 时，router 返回的 id 需要和 `ExpertMapManager` 的映射一致。

文档里可以把它理解为：

```text
router 选择的是逻辑/全局 expert；
expert_map 决定当前 rank 上哪些 expert 可见、如何映射到本地 expert slot。
```

---

## 7. `RoutedExperts` 如何保存 expert 权重

### 7.1 未量化权重布局

`UnquantizedFusedMoEMethod.create_weights()` 会注册：

```text
w13_weight: [num_experts, 2 * intermediate_size_per_partition, hidden_size]
w2_weight:  [num_experts, hidden_size, intermediate_size_per_partition]
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:88`

如果 `has_bias=True`，还会有：

```text
w13_bias
w2_bias
```

### 7.2 为什么是 w13

大多数 gated MLP 里 expert 有三组 projection：

```text
w1: gate_proj
w3: up_proj
w2: down_proj
```

vLLM 把 `w1` 和 `w3` 合并成 `w13`，这样第一段 GEMM 可以一次算出 gate/up 两部分，再接 fused activation。

### 7.3 weight_loader 需要处理 expert id 和 shard id

`RoutedExperts.weight_loader()` 处理 checkpoint 权重到本地 expert 参数的加载。

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:571`

它要同时考虑：

```text
- 当前 rank 是否持有这个 expert；
- w1 / w2 / w3 分别落到 w13 还是 w2；
- TP shard 如何切；
- quant scale / zero point 是否也按 expert 维度加载；
- redundant expert / EPLB 映射。
```

### 7.4 process_weights_after_loading 会准备 kernel 布局

未量化路径中，`process_weights_after_loading()` 会根据后端做：

- ROCm padding
- CPU packed weight
- XPU transpose
- kernel 格式转换
- 初始化 `moe_kernel`

位置：`code/vllm/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:201`

这一步很关键：

```text
forward 里不应该再做重量级 weight repack；
weight layout 应该在加载后处理好，forward 只做 routing 和 GEMM。
```

---

## 8. Modular kernel 路径如何执行 expert compute

### 8.1 runner 先选择 expert，再调用 RoutedExperts

位置：`code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:551`

逻辑是：

```text
if quant_method.is_monolithic:
    routed_experts.forward_monolithic(x, router_logits)
else:
    topk_weights, topk_ids = router.select_experts(...)
    routed_experts.forward_modular(x, topk_weights, topk_ids)
```

### 8.2 RoutedExperts 把执行委托给 quant_method

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:1053`

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

这意味着 MoE 的 kernel 选择并不由 `MoERunner` 写死，而是由 quant method / backend 决定。

### 8.3 未量化路径调用 `moe_kernel.apply()`

位置：`code/vllm/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:313`

关键参数包括：

```text
hidden_states
w1 / w13
w2
topk_weights
topk_ids
activation
apply_router_weight_on_input
global_num_experts
expert_map
shared_experts
shared_experts_input
```

其中 `expert_map` 用来处理 EP rank 不持有某些 expert 的情况。

---

## 9. functional `fused_experts()` 的经典 Triton 路径

虽然当前新接口逐步迁移到 modular kernel，但 `fused_moe.py` 里仍然清楚展示了传统 fused MoE 的三段式计算。

入口：`code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1474`

### 9.1 先准备 expert assignment

`_prepare_expert_assignment()` 会把 `topk_ids` 转成 kernel 更好消费的结构。

位置：`code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1425`

产物：

```text
sorted_token_ids
expert_ids
num_tokens_post_padded
```

典型逻辑是：

```text
1. 每个 token 会按 top_k 被重复；
2. token-expert 对按 expert 分组；
3. 每个 expert 的 token 数 padding 到 BLOCK_SIZE_M 对齐；
4. expert_ids 告诉每个 block 使用哪个 expert 的权重。
```

如果 token 数很小且 expert 很稀疏，也可能走 naive block assignment，跳过完整 alignment。

### 9.2 第一段 GEMM：hidden_states × w13

位置：`code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1672`

输入：

```text
hidden_states 或 quantized hidden_states
w1/w13
sorted_token_ids
expert_ids
topk_weights
```

输出到：

```text
intermediate_cache1: [num_tokens, top_k, intermediate * 2]
```

如果 `apply_router_weight_on_input=True`，top-k weight 会在第一段 GEMM 前或过程中应用；否则后面再应用。

### 9.3 fused activation

位置：`code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1696`

```python
apply_moe_activation(activation_enum, intermediate_cache2, intermediate_cache1.view(-1, N))
```

这一步把 gate/up 输出合成 MLP 中间激活，例如：

```text
SwiGLU: silu(gate) * up
GeGLU:  gelu(gate) * up
SwiGLU-OAI / clamped activation
```

输出到：

```text
intermediate_cache2: [num_tokens * top_k, intermediate]
```

### 9.4 第二段 GEMM：activation × w2

位置：`code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1711`

输出到：

```text
intermediate_cache3: [num_tokens, top_k, hidden]
```

### 9.5 combine：把 top-k expert 输出加回 token

位置：`code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1735`

```python
ops.moe_sum(intermediate_cache3, out_hidden_states)
```

底层 op 位置：`code/vllm/vllm/_custom_ops.py:2228`

语义是：

```text
对每个 token，把 top_k 个 expert 输出按权重合并成一个 hidden vector。
```

---

## 10. dispatch / combine 和并行通信

MoE 并行最容易混淆，因为它同时可能涉及 TP、EP、DP、PCP、SP。

### 10.1 naive DP/EP dispatch

位置：`code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:724`

如果：

```text
dp_size > 1 且 quant_method 不支持 internal modular kernel
```

则 runner 会调用：

```text
get_ep_group().dispatch_router_logits(hidden_states, router_logits, is_sequence_parallel)
```

把 token 和 router logits 分发到对应 EP rank。

### 10.2 combine

位置：`code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:757`

对应地，计算完成后会调用：

```text
get_ep_group().combine(hidden_states, is_sequence_parallel)
```

把各 rank 的结果合回去。

### 10.3 PCP all-gather / reduce-scatter

PCP 路径在 `_maybe_dispatch()` 和 `_maybe_combine()` 里也有单独处理：

```text
forward 前 all_gather hidden_states/router_logits
forward 后 reduce_scatter hidden_states
```

位置：

- `code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:745`
- `code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:767`

### 10.4 TP/EP all-reduce

MoE 输出可能需要 all-reduce，但时机取决于 kernel 是否已经 reduced。

相关逻辑：

- `_fused_output_is_reduced`: `code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:404`
- `_maybe_reduce_shared_expert_output`: `code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:411`
- `_maybe_reduce_final_output`: `code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:431`

可以记成：

```text
如果 combine kernel 已经 reduce fused_output，shared_output 单独 reduce；
否则先 shared + fused，再对最终结果 reduce。
```

---

## 11. 量化 MoE 如何改变执行路径

MoE 量化和普通 linear 量化类似，但多了 expert 维度和 top-k dispatch。

### 11.1 普通 linear 的量化接口

普通 linear：

```text
LinearBase
  → quant_method.create_weights()
  → quant_method.apply(x, bias)
```

### 11.2 MoE 的量化接口

MoE：

```text
RoutedExperts
  → quant_method.create_weights(num_experts, hidden, intermediate, ...)
  → quant_method.get_fused_moe_quant_config(layer)
  → quant_method.apply(layer, x, topk_weights, topk_ids, ...)
```

关键差异是：

```text
MoE quant method 必须知道 expert 维度、w13/w2 布局、expert_map、top-k ids、top-k weights、activation、combine 方式。
```

### 11.3 quant config 会传进 fused kernel

`fused_experts()` 里的 `FusedMoEQuantConfig` 会携带：

```text
use_fp8_w8a8
use_int8_w8a8
use_int8_w8a16
use_int4_w4a16
w1_scale / w2_scale
w1_zp / w2_zp
a1_scale / a2_scale
block_shape
bias
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1474`

这些字段会决定底层 GEMM 使用 FP8、INT8、INT4 还是未量化权重路径。

### 11.4 activation quantization 可能发生在两段 GEMM 前

在 functional path 中：

```text
hidden_states → moe_kernel_quantize_input() → qhidden_states
activation_out → moe_kernel_quantize_input() → qintermediate_cache2
```

位置：

- `code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1651`
- `code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1700`

也就是说，W8A8/FP8 类路径不仅改变 weight，还可能在 forward 中增加 activation quant kernel。

---

## 12. shared experts、zero expert 和 latent MoE

### 12.1 shared experts

`MoERunner` 可以带 `shared_experts`，并支持和 routed experts overlap。

相关位置：

- `code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:277`
- `code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:525`
- `code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:600`

结果合并逻辑：

```text
result = shared_output + fused_output
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:709`

### 12.2 zero expert

如果 router 是 `ZeroExpertRouter`，最终还会加上 zero expert 输出。

位置：`code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:612`

### 12.3 latent MoE transform

某些模型 routed experts 在 latent hidden dim 上计算，需要：

```text
routed_input_transform
routed_output_transform
```

相关位置：

- `code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:345`
- `code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:370`
- `code/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:466`

如果 hidden dim 被 padding 到 kernel 支持的对齐尺寸，runner 会在输出处再 truncate。

---

## 13. fused MoE 与 unfused fallback 的差异

可以从三层看差异。

### 13.1 routing 层

fused routing：

```text
router_logits → fused top-k op → topk_weights/topk_ids
```

unfused routing：

```text
softmax/sigmoid → torch.topk → gather/scatter → renormalize
```

fused routing 减少临时张量和 kernel launch。

### 13.2 expert compute 层

fused/grouped MoE：

```text
token-expert pair 按 expert 分组
每个 expert 的 token block 批量 GEMM
同一个 kernel grid 覆盖多个 expert
```

unfused MoE：

```text
for expert in experts:
    取出属于该 expert 的 token
    执行独立 MLP
    scatter add 回 output
```

unfused 路径通常会有更多 Python 调度、更多小 GEMM、更差的 GPU 利用率。

### 13.3 combine 层

fused combine：

```text
kernel 或 moe_sum 一次性完成 top-k reduce
```

unfused combine：

```text
scatter / index_add / for-loop add
```

当 batch 很小或 token 分布极不均衡时，fused kernel 也可能被 launch overhead 或 token imbalance 限制。

---

## 14. 性能瓶颈通常在哪里

### 14.1 token imbalance

如果大量 token 都路由到少数 expert：

```text
某些 expert block 很重，其他 expert block 很空，GPU 并行度下降。
```

表现：

```text
MoE kernel 时间波动大；
同样 token 数下不同 batch 延迟差异明显。
```

### 14.2 top-k 和 alignment 开销

小 batch 下，routing、排序、padding、`moe_align_block_size()` 的开销可能接近 GEMM 本身。

位置：`code/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py`

### 14.3 expert_map 导致无效 expert block

EP 下当前 rank 不持有某些 expert，kernel 可能需要写零或跳过。

在 Triton kernel 里，`off_experts == -1` 会写零输出。

位置：`code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:160`

### 14.4 quantized path 的 scale / activation quant 成本

FP8/INT8 W8A8 路径可能额外做 activation quant：

```text
hidden_states quant
intermediate activation quant
scale tensor load
zero point 处理
```

如果 batch 太小，这些额外 kernel 可能抵消量化 GEMM 的收益。

### 14.5 CUDA graph / torch.compile 约束

MoE 里有很多动态因素：

```text
topk_ids shape
expert token 分布
routing tables
workspace
shared expert overlap
LoRA active state
```

因此 vLLM 用 `torch.ops.vllm.moe_forward` 把 MoE 包成 custom op，并通过 layer registry / forward context 找回真实 layer。

---

## 15. 常见问题和排查

### 15.1 expert id 错或输出异常

优先检查：

```text
1. router 是否返回预期 topk_ids；
2. scoring_func / renormalize / grouped_topk 参数是否匹配模型；
3. ExpertMapManager 的 expert_map 是否和 EP 配置一致；
4. checkpoint expert id 到 local expert id 的映射是否正确；
5. w1/w2/w3 到 w13/w2 的加载是否正确。
```

### 15.2 shape 不匹配

常见原因：

```text
- hidden_size 没有按 backend 要求 round up；
- intermediate_size_per_partition 和 TP/EP 不整除；
- w13 的 2 * intermediate 维度和 activation 类型不匹配；
- quant group size / block_shape 和 expert weight shape 不匹配；
- latent MoE transform 前后 hidden dim 没有正确 truncate。
```

### 15.3 性能不符合预期

排查方向：

```text
- 看 batch token 数和 top_k；
- 看 token 是否集中到少数 expert；
- 看当前 moe_backend 选择；
- 看是否走了 monolithic / modular / fallback；
- 看 quant method 是否初始化了 moe_kernel；
- 看是否发生 DP/EP naive dispatch/combine；
- 关闭或开启 CUDA graph 对比。
```

### 15.4 量化 MoE 输出异常

优先检查：

```text
- w1_scale / w2_scale 是否按 expert 对齐；
- w1_zp / w2_zp 是否加载正确；
- w13 中 w1/w3 顺序是否符合 kernel 预期；
- activation quant 的 a1_scale / a2_scale 是否匹配；
- process_weights_after_loading 是否改变了 weight layout。
```

---

## 16. 最终可以记成一张表

| 阶段 | 主要对象 / 函数 | 核心产物 | 作用 |
|---|---|---|---|
| MoE 构造 | `FusedMoE()` | `MoERunner` | 创建 router、routed experts、runner |
| 并行配置 | `FusedMoEParallelConfig` | TP/DP/EP/SP 配置 | 决定 expert placement 和通信方式 |
| expert 映射 | `ExpertMapManager` | `expert_map`、routing tables | 把全局 expert 映射到本地 rank |
| routing | `FusedMoERouter.select_experts()` | `topk_weights`、`topk_ids` | 给每个 token 选择 top-k expert |
| 权重持有 | `RoutedExperts` | `w13_weight`、`w2_weight`、scale | 保存 expert 参数和量化参数 |
| 后端策略 | `FusedMoEMethodBase` | `moe_kernel`、`FusedMoEQuantConfig` | 决定权重布局和 kernel 调用 |
| expert compute | `quant_method.apply()` | `fused_output` | 执行 dispatch + grouped GEMM + activation + combine |
| shared expert | `SharedExperts` | `shared_output` | 处理 shared expert 分支 |
| 通信合并 | `_maybe_combine()` / `_maybe_reduce_final_output()` | final hidden states | 处理 EP/TP/PCP/SP 输出合并 |
| 底层 op | `fused_experts()` / `moe_sum` / native kernels | output tensor | 真正执行 fused MoE 计算 |

---

## 17. 最小心智模型

如果只记住一条主线：

```text
MoE forward = routing + expert grouped GEMM + weighted combine。
```

在 vLLM 里对应为：

```text
MoERunner 负责组织流程，
Router 负责 top-k，
RoutedExperts 负责权重和 quant_method，
quant_method / moe_kernel 负责真正执行 fused expert compute，
ExpertMapManager 和 runner 的 dispatch/combine 负责并行环境下的 expert placement 与通信。
```

再压缩成一句话：

```text
Fused MoE 的核心不是把所有 expert 写进一个 Python loop，
而是把 token-expert pair 排成适合 grouped GEMM 的布局，
用 fused kernel 连续完成 w13、activation、w2 和 top-k combine。
```
