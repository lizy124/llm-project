# 10. 哪些场景会导致 cudagraph fallback？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\config\compilation.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\config\vllm.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\config\parallel.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\cudagraph_dispatcher.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu\model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu\cudagraph_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\dp_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backend.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\encoder_cudagraph.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\wrapper.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\quantization\`

本问题关注：哪些配置、模型特性、输入形态、attention backend 能力、parallelism 组合或调试路径会导致 vLLM 放弃 CUDA Graph / torch compile 优化路径，fallback 到 piecewise graph 或 eager；以及 fallback 如何保证执行正确性。

---

## 1. 一句话回答

CUDA Graph 不是正确性路径，而是优化路径。

只要本轮执行不能满足：

```text
固定 shape、稳定 tensor 地址、稳定 kernel launch、稳定 attention metadata、稳定控制流、backend 支持 capture
```

vLLM 就会 fallback。

fallback 的层级通常是：

```text
FULL cudagraph
  → PIECEWISE cudagraph
  → NONE / eager forward
```

也就是说：

```text
能整图 replay 就整图 replay；
整图不安全就只 replay 编译子图；
子图也不安全或没命中 graph key，就普通 eager 执行。
```

最小记忆：

```text
cudagraph miss 不是异常；eager fallback 是语义兜底。
```

---

## 2. 本文要回答的问题

```text
哪些配置会直接禁用 cudagraph？
FULL / PIECEWISE / NONE 是如何降级的？
batch size 没命中 capture size 时怎么办？
为什么 prefill、mixed batch、spec decode 更容易 fallback？
attention backend 如何声明自己能不能 full cudagraph？
LoRA、multimodal、encoder-decoder、pooling、quantization 有哪些限制？
DP / TP / SP / DCP / DeepEP 会如何影响 cudagraph？
debug / profile / KV scale 计算为什么会强制 eager？
fallback 后如何保证 KV cache、attention metadata、sampling 语义正确？
```

---

## 3. 先给总图：fallback 发生在哪几层

vLLM 的 cudagraph fallback 不是单点判断，而是分布在 4 层。

```text
1. 配置归一层
   VllmConfig / CompilationConfig 根据模型、backend、环境变量、并行配置
   自动修正 cudagraph_mode / capture sizes。

2. capture key 层
   CudagraphDispatcher / CUDAGraphManager 预先决定哪些 BatchDescriptor 可以 capture。

3. 每轮 dispatch 层
   GPUModelRunner 根据本轮 batch shape、LoRA、encoder output、cascade attention、DP 对齐等
   选择 FULL / PIECEWISE / NONE。

4. wrapper 执行层
   CUDAGraphWrapper 或新版 manager 命中 key 就 replay；
   没命中、mode 为 NONE、或 skip_compiled=True 就调用原始 runnable。
```

完整链路可以写成：

```text
engine args / VllmConfig
  → CompilationConfig 校验和自动降级
  → attention backend 初始化后再次修正 cudagraph_mode
  → 初始化 cudagraph keys / capture sizes
  → warmup + capture 可捕获 graph
  → 每轮 execute_model()
      → _determine_batch_execution_and_padding()
      → dispatch FULL / PIECEWISE / NONE
      → set_forward_context(...)
      → _model_forward()
      → cudagraph replay 或 eager fallback
```

所以 fallback 可以发生在：

```text
还没开始运行时：配置层直接禁用；
初始化 capture 时：某些 key 根本不 capture；
每轮执行时：本轮 batch 不满足已捕获 key；
模型 forward 时：context 要求跳过 compiled path。
```

---

## 4. cudagraph mode 的基本语义

### 4.1 CUDAGraphMode 有运行时模式和组合模式

`CUDAGraphMode` 定义在：

```text
code/vllm/vllm/config/compilation.py:53
```

核心运行时模式是：

```text
NONE
PIECEWISE
FULL
```

配置层还支持组合模式：

```text
FULL_DECODE_ONLY
FULL_AND_PIECEWISE
```

可以这样理解：

| mode | 含义 |
|---|---|
| `NONE` | 不使用 cudagraph，普通 eager / compiled callable 执行 |
| `PIECEWISE` | 只对被 torch compile 切出来的子图做 cudagraph |
| `FULL` | 对整个 model forward 捕获 / replay |
| `FULL_DECODE_ONLY` | 只有 decode 类 batch 可以 full cudagraph，其他场景 fallback |
| `FULL_AND_PIECEWISE` | full 能用时用 full，不能用时退到 piecewise |

实际运行时每轮最终还是会落到：

```text
FULL / PIECEWISE / NONE
```

### 4.2 FULL 和 PIECEWISE 的差别

`FULL` 要求更高：

```text
整个 model forward 的输入 shape、attention metadata、控制流、kernel launch 都要稳定。
```

`PIECEWISE` 要求低一些：

```text
只要求被 compile 拆出来的局部子图满足 cudagraph replay 条件；
模型外层 Python 控制流、部分动态 metadata 可以留在 graph 外。
```

因此很多降级不是：

```text
FULL → eager
```

而是：

```text
FULL → PIECEWISE → eager
```

---

## 5. 配置层：哪些配置会直接禁用或降级 cudagraph

### 5.1 enforce_eager 直接禁用 compile 和 cudagraph

`VllmConfig` 初始化时，如果 `model_config.enforce_eager=True`，会直接禁用 compile 和 cudagraph。

位置：

```text
code/vllm/vllm/config/vllm.py:1071
```

语义是：

```text
enforce_eager 是用户明确要求 eager；
此时不应该再偷偷走 cudagraph replay。
```

结果通常是：

```text
compilation_config.mode = NONE
compilation_config.cudagraph_mode = NONE
```

### 5.2 TORCH_COMPILE_DISABLE=1 禁用 compile

如果环境变量禁用了 torch compile，vLLM 也会禁用相关 compile 路径。

位置：

```text
code/vllm/vllm/config/vllm.py:1079
```

这会影响 `PIECEWISE`，因为 piecewise cudagraph 依赖编译子图。

可以理解为：

```text
没有 compile 子图，就没有 piecewise graph 的基础。
```

### 5.3 VLLM_USE_BREAKABLE_CUDAGRAPH 会改变默认 compile pipeline

当开启 `VLLM_USE_BREAKABLE_CUDAGRAPH=1` 时，vLLM 会禁用自己的一部分 compile pipeline。

位置：

```text
code/vllm/vllm/config/vllm.py:1109
```

这类开关通常用于调试或实验路径，不应该和默认的 piecewise compile 语义混淆。

### 5.4 mode 不是 VLLM_COMPILE 时，依赖 piecewise 的模式会被降级

如果 `cudagraph_mode` 需要 piecewise，但当前 `compilation_config.mode != VLLM_COMPILE`，且没有使用 breakable cudagraph，那么会被降为 `NONE`。

位置：

```text
code/vllm/vllm/config/vllm.py:1182
```

原因是：

```text
PIECEWISE cudagraph 的 graph boundary 来自 vLLM compile pipeline；
没有 compile pipeline，就没有可捕获的 piecewise 子图。
```

### 5.5 pooling model 不支持 FULL cudagraph

pooling model 会强制把 full cudagraph 改成 piecewise。

位置：

```text
code/vllm/vllm/config/vllm.py:1243
```

原因是 pooling 输出形态和生成模型不同，通常更动态：

```text
生成模型：通常关注每轮 logits / sampling token；
pooling 模型：输出可能和请求级 pooling 位置、pooling method、序列长度有关。
```

所以配置层直接避免 full graph：

```text
FULL → PIECEWISE
```

### 5.6 encoder-decoder 只允许 NONE 或 FULL_DECODE_ONLY

encoder-decoder 模型的 cudagraph 模式会被限制。

位置：

```text
code/vllm/vllm/config/vllm.py:1255
```

原因是 encoder-decoder 有两类动态：

```text
encoder input / encoder output 动态；
cross attention metadata 动态。
```

所以不适合无条件 full graph。

常见语义是：

```text
带 encoder input / encoder output 的步骤：跳过 compiled/full graph；
纯 decode 且条件满足时：可以尝试 FULL_DECODE_ONLY。
```

### 5.7 某些 KV connector 要求 PIECEWISE

部分 KV connector 和 full cudagraph 不兼容，会把 full 模式降到 piecewise。

位置：

```text
code/vllm/vllm/config/vllm.py:1269
```

原因是 KV connector 可能引入：

```text
外部 KV 传输；
动态 metadata；
跨层或跨请求的同步状态；
不适合整图 capture 的 Python 控制流。
```

所以更安全的策略是：

```text
FULL → PIECEWISE
```

### 5.8 eager 下 capture sizes 会被清空

如果最终不使用 cudagraph，配置层会清空：

```text
max_cudagraph_capture_size = 0
cudagraph_capture_sizes = []
```

位置：

```text
code/vllm/vllm/config/vllm.py:1296
code/vllm/vllm/config/vllm.py:1804
```

这很重要：

```text
后续 dispatcher 看到空 capture sizes，就不会再试图 dispatch 到 graph。
```

---

## 6. attention backend 层：backend 能力会自动降级 cudagraph

### 6.1 attention backend 会声明 full cudagraph 支持等级

attention backend 通过 `AttentionCGSupport` 表达自己的 cudagraph 能力。

位置：

```text
code/vllm/vllm/v1/attention/backend.py:516
```

常见等级：

```text
ALWAYS
UNIFORM_BATCH
UNIFORM_SINGLE_TOKEN_DECODE
NEVER
```

含义：

| 支持等级 | 含义 |
|---|---|
| `ALWAYS` | backend 认为自己可以支持 full cudagraph |
| `UNIFORM_BATCH` | 只支持 uniform batch 的 full graph |
| `UNIFORM_SINGLE_TOKEN_DECODE` | 只支持普通单 token decode |
| `NEVER` | 不支持 full cudagraph |

### 6.2 backend 不支持 mixed full graph 时会降级

`resolve_cudagraph_mode_and_sizes()` 会根据 backend 能力修正 `cudagraph_mode`。

位置：

```text
code/vllm/vllm/config/compilation.py:1316
```

典型规则：

```text
如果 backend 不支持 mixed batch full cudagraph：
  FULL → FULL_AND_PIECEWISE 或 FULL_DECODE_ONLY

如果 backend 连 decode full cudagraph 也不支持：
  FULL → PIECEWISE 或 NONE
```

也就是说，用户配置的 mode 只是目标值，最终值还要经过 backend 能力裁剪。

### 6.3 metadata 不支持 capture 会阻止 full graph

attention backend metadata 约束入口在：

```text
code/vllm/vllm/v1/attention/backend.py:292
```

典型不支持原因包括：

```text
partial multimodal token full attention
per-head quant scales
batch invariance 不满足
KV connector 不支持
MLA / sparse / sink / non-causal / block_size / dtype / attn_type 限制
```

原因是 full cudagraph 不只 capture linear 层，它还 capture attention kernel。

attention kernel 是否可 replay 依赖：

```text
query_start_loc
seq_lens
block_table
slot_mapping
max_query_len
max_seq_len
backend wrapper / plan
```

这些 metadata 如果每轮结构不同，就不能安全 replay 同一个 graph。

### 6.4 DCP 和 spec decode 会改变 metadata 约束

metadata builder 对 speculative decode 和 DCP 有特殊处理。

位置：

```text
code/vllm/vllm/v1/attention/backend.py:567
```

典型逻辑：

```text
支持 spec-as-decode 的 backend：
  可以把 reorder_batch_threshold 提高到 spec token 相关阈值。

DCP > 1 且 backend 不支持 varlen DCP：
  强制 reorder_batch_threshold = 1。
```

这说明：

```text
spec decode / DCP 不是简单扩大 batch size；
它会改变 attention metadata 的合法结构。
```

---

## 7. capture size 层：batch shape 不在捕获集合里会 fallback

### 7.1 capture sizes 是 cudagraph 可 replay 的 batch 档位

`cudagraph_capture_sizes` 决定哪些 token 数 / batch size 会被 capture。

默认生成逻辑在：

```text
code/vllm/vllm/config/vllm.py:1689
```

默认 `max_cudagraph_capture_size` 大致受这些值限制：

```text
max_num_seqs
num_speculative_tokens
max_num_batched_tokens
```

如果用户没有显式配置 capture sizes，vLLM 会生成一组离散档位，例如：

```text
小 batch 更密集；
大 batch 按 8 或 16 的步长增长；
最大不超过 max_cudagraph_capture_size。
```

### 7.2 运行时 batch size 会 pad 到下一档 capture size

旧版 `CudagraphDispatcher` 会预计算：

```text
真实 batch size → padded graph size
```

位置：

```text
code/vllm/vllm/v1/cudagraph_dispatcher.py:72
```

规则是：

```text
如果真实 size 命中 capture size：
  不 padding。

如果真实 size 落在两个 capture size 中间：
  pad 到下一个 capture size。
```

举例：

```text
capture sizes = [1, 2, 4, 8]
真实 num_tokens = 5
→ padded graph size = 8
```

这样可以复用已 capture 的 graph。

### 7.3 超过 max_cudagraph_capture_size 直接 NONE

如果本轮：

```text
num_tokens > max_cudagraph_capture_size
```

dispatcher 会直接返回 `NONE`。

位置：

```text
code/vllm/vllm/v1/cudagraph_dispatcher.py:274
```

原因很简单：

```text
没有对应或更大的 captured graph 可以 replay；
为了正确性只能 eager。
```

### 7.4 key 未初始化或 mode 为 NONE 直接 NONE

dispatcher 还有几个直接 fallback 条件：

```text
cudagraph keys 尚未初始化；
cudagraph_mode == NONE；
可选模式集合里已经没有 FULL / PIECEWISE。
```

位置：

```text
code/vllm/vllm/v1/cudagraph_dispatcher.py:274
```

这类 fallback 通常不是输入问题，而是初始化或配置已经决定不用 cudagraph。

### 7.5 compile_sizes 不能落在会被 padding 改写的位置

`compile_sizes` 和 cudagraph padding 有一致性约束。

位置：

```text
code/vllm/vllm/v1/cudagraph_dispatcher.py:93
```

如果某个 `compile_size` 会被 padding 到另一个 capture size，vLLM 会直接报错。

原因是：

```text
compile 子图 shape 和 cudagraph replay shape 必须一致；
否则 compile 以为自己编译的是 A，runtime 却拿 B 的 graph replay。
```

---

## 8. 每轮 dispatch：GPUModelRunner 如何选择 FULL / PIECEWISE / NONE

### 8.1 主决策点在 _determine_batch_execution_and_padding()

旧版主线在：

```text
code/vllm/vllm/v1/worker/gpu_model_runner.py:3810
```

这一段每轮会做：

```text
1. 判断本轮是否 uniform decode；
2. 判断 encoder-decoder 当前步是否有 encoder output；
3. 统计当前 active LoRA；
4. 先做 sequence parallelism padding；
5. 调 CudagraphDispatcher.dispatch()；
6. DP 模式下跨 rank 同步 mode 和 padding；
7. 返回 cudagraph_runtime_mode 和 BatchDescriptor。
```

对应主线：

```text
execute_model()
  → _prepare_inputs()
  → _determine_batch_execution_and_padding()
  → _get_slot_mappings()
  → _build_attention_metadata()
  → set_forward_context(...)
  → _model_forward()
```

### 8.2 uniform decode 是 FULL 的重要前提

`uniform_decode` 的直觉含义是：

```text
所有请求本轮执行相同数量的 query token。
```

普通 decode 时通常是：

```text
每个 request 1 token
```

speculative decode 验证步可能是：

```text
每个 request 1 + num_speculative_tokens token
```

如果 batch 里混了 prefill 和 decode，或者不同请求 query length 不一致，就更难 full graph。

### 8.3 use_cascade_attn 会禁用 FULL

`GPUModelRunner` dispatch 时会把 `use_cascade_attn` 作为禁用 full graph 的条件。

位置：

```text
code/vllm/vllm/v1/worker/gpu_model_runner.py:3865
```

原因是 cascade attention 会改变 attention 执行结构：

```text
prefix / suffix 可能走不同 attention path；
metadata 和 kernel launch 不一定和普通 full graph 一致。
```

因此策略是：

```text
禁用 FULL；
如果 PIECEWISE 可用，退到 PIECEWISE；
否则 NONE。
```

### 8.4 encoder output 存在时会禁用 FULL

encoder-decoder 当前步如果带 `encoder_output`，也会禁用 full graph。

位置：

```text
code/vllm/vllm/v1/worker/gpu_model_runner.py:3839
code/vllm/vllm/v1/worker/gpu_model_runner.py:3865
```

原因是 encoder output / cross attention metadata 形态动态：

```text
不同请求 encoder length 不同；
encoder output tensor 可能每轮变化；
cross attention 的 max_seqlen_k / metadata 也会变化。
```

### 8.5 calculate_kv_scales 会强制 NONE

如果本轮需要计算 KV scale，旧版 runner 会强制：

```text
CUDAGraphMode.NONE
```

位置：

```text
code/vllm/vllm/v1/worker/gpu_model_runner.py:4282
```

原因是 KV scale 计算通常是初始化 / profile / 量化辅助路径：

```text
它不是稳定的普通 decode forward；
它可能修改运行时状态；
不应该被 capture 到可反复 replay 的 graph 里。
```

### 8.6 encoder-decoder 有 encoder input 时跳过 compiled path

encoder-decoder 且本轮有 encoder input 时，会跳过 compiled path。

位置：

```text
code/vllm/vllm/v1/worker/gpu_model_runner.py:4290
```

这和前面的 full graph 限制是一致的：

```text
带 encoder input 的步骤更动态，优先保证正确性。
```

### 8.7 新版 GPU runner 显式分流 FULL / PIECEWISE / NONE

新版路径在：

```text
code/vllm/vllm/v1/worker/gpu/model_runner.py:1257
```

分流语义是：

```text
FULL      → run_fullgraph()
PIECEWISE → run_pw_graph()
NONE      → self.model(**model_inputs)
```

位置：

```text
code/vllm/vllm/v1/worker/gpu/model_runner.py:1257
code/vllm/vllm/v1/worker/gpu/model_runner.py:1283
code/vllm/vllm/v1/worker/gpu/model_runner.py:1291
```

也就是说新版代码把 fallback 表达得更直接：

```text
没有 graph，就直接调用普通 model forward。
```

---

## 9. BatchDescriptor：为什么 key miss 会 fallback

### 9.1 BatchDescriptor 是 cudagraph dispatch 的 key

cudagraph 不是只按 `num_tokens` 查表，还会考虑 batch 语义。

典型 key 包含：

```text
num_tokens
num_reqs
uniform
has_lora
num_active_loras
```

含义：

```text
同样 num_tokens，如果 request 数不同，attention metadata 可能不同；
同样 batch size，如果 LoRA active 数不同，LoRA kernel / mapping 可能不同；
同样 token 数，如果 uniform decode 和 mixed batch 不同，attention routine 可能不同。
```

### 9.2 dispatch 优先级是 FULL → PIECEWISE → NONE

旧版 dispatcher 的调度优先级：

```text
先尝试 FULL key；
FULL 不存在再尝试 PIECEWISE key；
都不存在就 NONE。
```

位置：

```text
code/vllm/vllm/v1/cudagraph_dispatcher.py:307
```

这也是 fallback 的核心：

```text
不是配置里有 FULL 就一定 FULL；
必须本轮 BatchDescriptor 命中 FULL key。
```

### 9.3 padding 后的 descriptor 必须和 capture descriptor 一致

如果真实 batch 被 pad 到某个 capture size，那么：

```text
input_ids / positions / slot_mapping / attention metadata
```

都要按 padded shape 构造。

否则会出现：

```text
graph replay 期待 shape = 8；
实际 metadata 只有 shape = 5；
```

这会破坏 replay 安全性。

因此 `_determine_batch_execution_and_padding()` 后面会影响：

```text
_get_slot_mappings()
_build_attention_metadata()
_preprocess()
set_forward_context()
```

---

## 10. attention metadata：为什么它是 fallback 的高发点

### 10.1 CUDA Graph capture 不只要求输入 tensor shape 固定

对 attention 来说，下面这些也要稳定：

```text
query_start_loc
query_start_loc_cpu
seq_lens
block_table_tensor
slot_mapping
max_query_len
max_seq_len
num_reqs
positions
encoder_seq_lens
dcp_local_seq_lens
backend metadata object / wrapper / plan
```

这些字段由 `_build_attention_metadata()` 构造，并交给具体 `AttentionMetadataBuilder`。

### 10.2 padding token 不能污染 KV cache

full cudagraph 下，padding token 也会进入固定 shape 的 graph。

为了正确性：

```text
padding token 的 slot_mapping 要填 -1；
padding request 的 block_table 要填 NULL_BLOCK_ID；
无效 token 的 logits / sampling 位置不能被当成真实输出。
```

否则 replay 时可能把 padding token 写进真实 KV cache。

### 10.3 max_query_len 会决定 attention routine

`max_query_len` 直接影响 attention 走 decode、spec decode 还是 prefill/mixed path。

```text
普通 decode：max_query_len = 1
spec decode：max_query_len = 1 + num_speculative_tokens
prefill / mixed：max_query_len = 本轮最长 query chunk
```

如果 capture 时是 decode，replay 时变成 mixed prefill，就不能复用同一个 full graph。

### 10.4 backend capture metadata 有专门入口

attention backend 有 cudagraph capture 专用 metadata 构造入口。

位置：

```text
code/vllm/vllm/v1/attention/backend.py:634
```

这说明：

```text
capture 阶段 metadata 不是普通 metadata 的无脑复用；
backend 需要显式声明自己怎么构造可 replay 的 metadata。
```

---

## 11. LoRA：active adapter 数动态会影响 graph key

### 11.1 LoRA 会进入 cudagraph key

LoRA 的运行时状态至少包括：

```text
has_lora
num_active_loras
```

这些会影响 cudagraph dispatch。

位置：

```text
code/vllm/vllm/v1/cudagraph_dispatcher.py:111
```

原因是：

```text
不同 active LoRA 数可能对应不同 LoRA mapping、不同 kernel 参数、不同权重选择路径。
```

### 11.2 specialize_lora 决定是否按 active count 细分 graph

配置项：

```text
cudagraph_specialize_lora
```

如果开启：

```text
会为多个 active LoRA count 捕获不同 graph key。
```

如果关闭：

```text
不为每个 active count 精确 capture；
通常把“有 LoRA”的情况映射到一个通用 case。
```

### 11.3 运行时 active LoRA 不一定精确命中

旧 dispatcher 会把真实 active LoRA 数映射到已 capture 的 case。

位置：

```text
code/vllm/vllm/v1/cudagraph_dispatcher.py:283
```

新版 manager 也有类似预计算映射。

位置：

```text
code/vllm/vllm/v1/worker/gpu/cudagraph_utils.py:157
```

可以总结为：

```text
LoRA 不是任意 active-count 都精确 capture；
运行时会复用或钳制到最近的已捕获 case；
没有合适 key 时继续 fallback。
```

---

## 12. multimodal / encoder cudagraph：视觉输入动态会触发 fallback

### 12.1 多模态 encoder 有单独的 cudagraph 配置

相关配置项在 `CompilationConfig` 中：

```text
compile_mm_encoder
cudagraph_mm_encoder
encoder_cudagraph_token_budgets
encoder_cudagraph_max_vision_items_per_batch
encoder_cudagraph_max_frames_per_batch
```

位置：

```text
code/vllm/vllm/config/compilation.py:516
```

原因是 multimodal encoder 和 decoder forward 的动态性不同：

```text
图像数量动态；
每张图 token 数动态；
视频 frame 数动态；
encoder 输出 shape 动态；
不同 processor / model 的视觉分支不同。
```

所以它不能简单套用 decoder 的 capture sizes。

### 12.2 encoder graph 有 token budget 限制

encoder cudagraph 会按 token budget 捕获不同 graph。

如果找不到合适预算，就会 eager fallback。

位置：

```text
code/vllm/vllm/v1/worker/encoder_cudagraph.py:341
code/vllm/vllm/v1/worker/encoder_cudagraph.py:469
```

含义：

```text
当前 multimodal 输入超过已捕获 token budget；
或者 vision items / frames 不满足预算；
则不 replay encoder graph，直接普通执行 encoder。
```

### 12.3 dual-path encoder 两路预算都不满足时整体 eager

某些 encoder 有 global/local 双路径。

如果 global 和 local 两路预算都不能满足，会整体 fallback 到 eager。

位置：

```text
code/vllm/vllm/v1/worker/encoder_cudagraph.py:598
```

这类 fallback 是为了避免：

```text
一部分 encoder 子路径 replay graph，另一部分 shape 不匹配。
```

### 12.4 partial multimodal token full attention 可能不支持

attention backend 可能拒绝 partial multimodal token full attention。

位置：

```text
code/vllm/vllm/v1/attention/backend.py:302
```

原因是部分多模态 token 可能需要非标准 attention mask / bidirectional range / document range。

这会破坏 full graph 对稳定 attention metadata 的要求。

---

## 13. speculative decode：query length 和 capture size 都会变

### 13.1 spec decode 的 uniform query length 不是 1

普通 decode：

```text
uniform_decode_query_len = 1
```

spec decode：

```text
uniform_decode_query_len = 1 + num_speculative_tokens
```

这会影响：

```text
num_tokens 如何换算成 num_reqs；
哪些 capture sizes 是合法的；
full decode graph 的 BatchDescriptor 如何生成。
```

### 13.2 capture sizes 会 round 到 spec query length 的倍数

spec decode 下，capture sizes 会被调整成：

```text
1 + num_speculative_tokens
```

的倍数。

位置：

```text
code/vllm/vllm/config/compilation.py:1421
```

如果开了 sequence parallelism，还要满足 TP 相关倍数约束。

位置：

```text
code/vllm/vllm/config/compilation.py:1462
```

如果 round 完没有合法 size，会直接报错，而不是运行时随机 fallback。

### 13.3 backend 不支持 full spec batch 时会自动降级

如果 attention backend 对 full decode 的 spec batch 支持不足，会降为：

```text
PIECEWISE
```

或：

```text
NONE
```

位置：

```text
code/vllm/vllm/config/compilation.py:1387
```

### 13.4 full-CG spec decode 仍有特殊 unpad 路径

`CommonAttentionMetadata.unpadded()` 附近有注释说明 full-CG spec decode 仍有专门处理。

位置：

```text
code/vllm/vllm/v1/attention/backend.py:479
```

这说明：

```text
spec decode 对 full cudagraph 仍然是高复杂度场景；
不是所有 backend / batch 形态都能无条件 replay。
```

---

## 14. quantization：大多不改变语义，但可能限制 graph

### 14.1 量化通常不改变 TP/forward 语义

量化主要改变：

```text
权重存储格式；
scale / zero point；
packing；
GEMM kernel；
attention quant kernel。
```

它不必然导致 cudagraph fallback。

### 14.2 per-head quant scales 可能被 attention backend 拒绝

attention backend 可能直接声明不支持某类 per-head quant scales 的 cudagraph。

位置：

```text
code/vllm/vllm/v1/attention/backend.py:318
```

原因是 per-head scale 可能引入额外动态 metadata 或 backend 不支持的 capture 形态。

### 14.3 fuse_attn_quant 和 piecewise cudagraph 有兼容性限制

当 `fuse_attn_quant` 和 piecewise cudagraph 配合，但 `use_inductor_graph_partition=False` 时，会强制改为 full。

位置：

```text
code/vllm/vllm/config/compilation.py:1204
```

可以理解为：

```text
某些 fused quant attention 路径不能安全地被切成 piecewise graph；
要么走 full，要么进一步被 backend 能力降级。
```

---

## 15. pooling：输出形态动态，FULL 会被禁用

pooling model 在配置层会禁用 full cudagraph。

位置：

```text
code/vllm/vllm/config/vllm.py:1243
```

pooling runner 目前也有模型类型限制。

位置：

```text
code/vllm/vllm/v1/worker/gpu/pool/pooling_runner.py:15
```

原因是 pooling 和生成式 decode 的输出不同：

```text
生成式 decode：通常只需要最后若干 logits；
pooling：需要根据 pooling method 从序列中抽取或聚合 embedding；
不同请求 pooling 位置和输出 shape 更动态。
```

所以策略是：

```text
不要 FULL；
能 PIECEWISE 就 PIECEWISE；
否则 eager。
```

---

## 16. parallelism：DP / TP / SP / DCP / DeepEP 都会影响 fallback

### 16.1 DP 下所有 rank 的 cudagraph mode 需要统一

DP 场景下，不同 DP rank 可能本轮 batch shape 不同。

但如果存在跨 DP 的 collective 或需要同步生成，不能让：

```text
rank0 走 FULL
rank1 走 NONE
```

否则可能出现 collective 顺序不一致或死锁。

因此 DP utils 会同步 cudagraph mode，通常取更保守的模式。

位置：

```text
code/vllm/vllm/v1/worker/dp_utils.py:92
code/vllm/vllm/v1/worker/dp_utils.py:139
```

直觉是：

```text
只要一个 rank 不能 graph，全体都可能要降级。
```

### 16.2 DP 下 token 数也可能需要对齐 padding

如果开启 cudagraph 或 ubatching，DP ranks 之间会 pad 到一致 token 数。

位置：

```text
code/vllm/vllm/v1/worker/dp_utils.py:147
```

原因是：

```text
跨 rank collective 要求参与方 shape 一致；
cudagraph replay 也要求每个 rank 的 launch 形态稳定。
```

### 16.3 sequence parallelism / async TP 可能影响 piecewise

当：

```text
use_inductor_graph_partition=False
```

且开启 sequence parallelism / async TP 相关路径时，vLLM 可能清空 `splitting_ops`，并把依赖 piecewise 的模式强制转成 `FULL`。

位置：

```text
code/vllm/vllm/config/compilation.py:1167
```

原因是：

```text
piecewise cudagraph 依赖 graph partition；
如果不允许 partition，就不能继续假装有 piecewise boundary。
```

### 16.4 DeepEP high-throughput + DP 会禁用 CUDA Graphs

`deepep_high_throughput + data_parallel_size > 1` 会直接禁用 CUDA graphs。

位置：

```text
code/vllm/vllm/config/compilation.py:1186
```

原因是 DeepEP all2all / expert parallel 通信和 cudagraph replay 的交互更复杂：

```text
通信 buffer；
all2all 调度；
DP rank 间 token 分布；
MoE expert routing 动态性。
```

### 16.5 DCP 有独立的并行合法性约束

DCP 配置约束在：

```text
code/vllm/vllm/config/parallel.py:490
```

典型约束：

```text
tensor_parallel_size 必须能被 decode_context_parallel_size 整除；
dcp_comm_backend='a2a' 要求 decode_context_parallel_size > 1。
```

DCP 会改变每个 rank 的 context shard 和 attention metadata，因此也会影响 cudagraph 能力判断。

### 16.6 MoE sequence parallel 是特殊优化，不等同普通 SP 入参

`ParallelConfig.use_sequence_parallel_moe` 说明：当启用 expert parallel、某些 all2all backend、TP>1、DP>1 时，会为了避免重复专家计算而使用 sequence-parallel MoE 输入。

位置：

```text
code/vllm/vllm/config/parallel.py:625
```

这不是一个普通 `sequence_parallel_size` 维度，而是 MoE + EP/DP/TP 下的执行策略。

它会影响：

```text
expert 输入是否 replicated；
all2all 前后的 token layout；
piecewise graph 是否能稳定分割。
```

---

## 17. debug / profile / instrumentation：调试路径会禁用或绕开 graph

### 17.1 layerwise NVTX tracing 和 cudagraph 不完全兼容

旧版 runner 对 layerwise NVTX tracing 有限制说明。

位置：

```text
code/vllm/vllm/v1/worker/gpu_model_runner.py:3930
```

原因是 cudagraph replay 后：

```text
很多 kernel launch 不再按普通 eager Python 调用栈出现；
逐层 hook / tracing 不一定完整或准确。
```

所以调试时经常需要 eager。

### 17.2 STOCK_TORCH_COMPILE 下不会注册 layerwise NVTX hooks

位置：

```text
code/vllm/vllm/v1/worker/gpu_model_runner.py:3941
```

这属于 instrumentation 限制：

```text
不是模型不能跑；
而是某些调试统计无法和 compile/cudagraph 同时保持准确。
```

### 17.3 compiled forward 中修改 buffer 可能直接报错

`compilation/wrapper.py` 中，如果 compiled forward 修改了 `nn.Module` buffer，且 cudagraph 开启，会直接抛错。

位置：

```text
code/vllm/vllm/compilation/wrapper.py:250
```

原因是：

```text
CUDA Graph replay 假设捕获的内存地址和状态使用方式稳定；
forward 内修改 module buffer 会让 capture/replay 的状态语义变得危险。
```

这类问题通常不是 fallback，而是明确报错提醒模型实现不满足 graph 假设。

---

## 18. fallback 场景总表

| 场景 | 触发条件 | fallback 行为 | 关键位置 |
|---|---|---|---|
| 显式 eager | `enforce_eager=True` | 禁用 compile/cudagraph | `config/vllm.py:1071` |
| torch compile 被禁用 | `TORCH_COMPILE_DISABLE=1` | 禁用 compile，piecewise 不可用 | `config/vllm.py:1079` |
| compile mode 不支持 piecewise | `mode != VLLM_COMPILE` | 依赖 piecewise 的 cudagraph 降为 NONE | `config/vllm.py:1182` |
| pooling model | pooling 不支持 full graph | `FULL → PIECEWISE` | `config/vllm.py:1243` |
| encoder-decoder | encoder/cross-attn 动态 | 只允许 `NONE/FULL_DECODE_ONLY` 或运行时 skip | `config/vllm.py:1255` |
| KV connector | connector 不支持 full graph | `FULL → PIECEWISE` | `config/vllm.py:1269` |
| attention backend 不支持 full | `AttentionCGSupport` 限制 | `FULL → PIECEWISE/NONE` | `config/compilation.py:1316` |
| 超出 capture size | `num_tokens > max_cudagraph_capture_size` | `NONE` | `v1/cudagraph_dispatcher.py:274` |
| batch key miss | `BatchDescriptor` 未捕获 | `FULL → PIECEWISE → NONE` | `v1/cudagraph_dispatcher.py:307` |
| cascade attention | `use_cascade_attn=True` | 禁用 FULL | `gpu_model_runner.py:3865` |
| encoder output | 当前步有 encoder output | 禁用 FULL / skip compiled | `gpu_model_runner.py:3839` |
| KV scale 计算 | `calculate_kv_scales=True` | 强制 NONE | `gpu_model_runner.py:4282` |
| LoRA active count 动态 | LoRA case 未精确捕获 | 映射到已捕获 case，否则 fallback | `cudagraph_dispatcher.py:283` |
| multimodal encoder budget miss | 找不到合适 encoder graph | encoder eager fallback | `encoder_cudagraph.py:341` |
| spec decode size 不合法 | capture size 不能满足 query len 倍数 | 报错或降级 | `compilation.py:1421` |
| per-head quant scales | backend 不支持 | 禁用对应 graph | `attention/backend.py:318` |
| DeepEP + DP | high-throughput backend + DP | 禁用 CUDA graphs | `compilation.py:1186` |
| DP rank 不一致 | 任一 rank 不能 graph | 全体取保守 mode | `dp_utils.py:92` |
| buffer mutation | compiled forward 修改 buffer | 抛错 | `compilation/wrapper.py:250` |

---

## 19. fallback 后如何保证正确性

fallback 的核心目标不是快，而是：

```text
输出语义一致；
KV cache 写入一致；
attention metadata 正确；
logits / sampler 不受影响；
collective 顺序不死锁；
不因为 graph miss 中断请求。
```

### 19.1 shape 不匹配时 eager 重新按真实 shape 执行

如果没有 graph，普通 forward 会按本轮真实输入执行：

```text
input_ids / positions / slot_mapping / attention metadata
```

都按真实 batch 构造，不需要强行套用 padded capture shape。

### 19.2 padding token 会被显式隔离

如果为了复用 graph 做了 padding：

```text
padding token 不应该写 KV cache；
padding request 不应该读真实 block；
padding logits 不应该参与采样。
```

因此相关字段会使用：

```text
slot_mapping = -1
block_table = NULL_BLOCK_ID
logits_indices 只指向真实 token
```

### 19.3 DP 先统一 mode，避免 rank 间分叉

DP 下如果各 rank mode 不一致，可能导致 collective 顺序不同。

所以会先同步：

```text
每个 rank 本轮能用什么 cudagraph mode；
本轮需要 pad 到多少 token。
```

再执行 forward。

### 19.4 cudagraph 是透明优化，不改变 sampler 语义

不管 forward 是：

```text
FULL replay
PIECEWISE replay
eager
```

最终都要产生相同语义的：

```text
hidden_states
logits
sampling metadata
ModelRunnerOutput
```

采样阶段不应该知道“前面是不是 graph replay”。

---

## 20. 常见疑问

### 20.1 batch size 没命中 capture size，一定 eager 吗？

不一定。

如果真实 size 小于等于最大 capture size，通常会：

```text
pad 到下一档 capture size
```

然后 replay 对应 graph。

只有当：

```text
超过最大 capture size；
或 padded 后的 BatchDescriptor 没有合法 key；
或 runtime feature 禁用了 graph；
```

才会 fallback 到 `NONE`。

### 20.2 prefill 为什么比 decode 更容易 fallback？

因为 prefill 的动态性更强：

```text
每个请求 query length 不同；
max_query_len 更大且变化；
block_table / seq_lens / query_start_loc 更动态；
可能混合 decode request；
attention kernel 可能走不同 routine。
```

普通 decode 更容易满足：

```text
每个 request 1 token；
shape 小且稳定；
metadata 结构相似。
```

### 20.3 FULL 不可用时，为什么 PIECEWISE 可能还可用？

因为 FULL 要 capture 整个 model forward。

PIECEWISE 只 capture 局部 compile 子图：

```text
动态 attention metadata、Python 控制流、部分 preprocessing
可以留在 graph 外。
```

所以它的适用范围更广，但收益也通常比 full graph 小。

### 20.4 fallback 会影响输出吗？

理论上不应该。

fallback 只改变执行路径：

```text
replay captured graph
vs
调用普通 forward
```

不改变模型数学语义。

如果 fallback 后输出不一致，通常说明：

```text
padding mask；
KV cache slot_mapping；
attention metadata；
LoRA mapping；
DP rank 对齐；
```

某处存在 bug。

### 20.5 cudagraph fallback 是错误吗？

不是。

正常 serving 中会频繁出现：

```text
大 prefill eager；
小 decode cudagraph；
某些 multimodal encoder eager；
某些 batch PIECEWISE；
某些 batch FULL。
```

这是预期的混合执行模式。

---

## 21. 最终可以记成一张图

```text
配置层：
  enforce_eager / env / compile mode / model type / kv connector / backend capability
    → 修正 cudagraph_mode

初始化层：
  capture sizes / LoRA cases / full vs piecewise descriptors
    → 生成可捕获 BatchDescriptor 集合

运行时：
  num_tokens / num_reqs / uniform decode / LoRA / encoder output / cascade attn / DP 对齐
    → dispatch FULL / PIECEWISE / NONE

执行层：
  FULL      → replay full graph
  PIECEWISE → replay compiled subgraph
  NONE      → eager model forward
```

一句话总结：

```text
vLLM 的 cudagraph fallback 本质是“先按配置和 backend 能力确定可用优化上限，再按每轮 batch descriptor 尝试命中 graph；命中就 replay，没命中就 eager，始终以 correctness 为底线”。
```
