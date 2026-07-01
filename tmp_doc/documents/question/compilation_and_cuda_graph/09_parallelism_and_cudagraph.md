# 09. 并行机制如何影响 CUDA graph / compile？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\config\parallel.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\config\compilation.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\config\vllm.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\forward_context.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\cudagraph_dispatcher.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_worker.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\dp_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\ubatch_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_ubatch_wrapper.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\cp_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\cuda_graph.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\backends.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\wrapper.py`

本问题关注：Tensor Parallel、Pipeline Parallel、Data Parallel、Expert Parallel、Sequence Parallel、Context Parallel、Dual Batch Overlap / microbatching 等并行机制，如何影响 vLLM V1 中的 torch.compile、CUDA graph capture / replay、batch padding、通信顺序和 fallback。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录文档风格，本篇按“先讲总原则，再逐个并行维度展开，最后给典型组合和排查方法”的方式梳理。

要回答的问题分成 14 组：

```text
1. 并行为什么会影响 CUDA graph？
2. Tensor Parallel 对 shape、collective、SP padding 有什么影响？
3. Sequence Parallel / async TP 为什么会改变 cudagraph sizes？
4. Pipeline Parallel 下每个 PP rank 捕获的 graph 为什么不同？
5. PP 中 intermediate_tensors 和 all-gather 如何影响 graph 形态？
6. Data Parallel 下为什么要跨 rank 协调 padding 和 cudagraph mode？
7. DPMetadata 如何进入 ForwardContext？
8. DBO / microbatching 如何拆分 metadata、slot_mapping 和 graph？
9. Expert Parallel / MoE / All2All 为什么会影响 CUDA graph 兼容性？
10. Context Parallel / DCP / PCP 对 attention metadata 有什么约束？
11. LoRA / routed experts / external launcher 等并行相关特性有什么边界？
12. V2 ModelRunner 对并行 + cudagraph 有哪些 unsupported？
13. capture / replay 时哪些并行信息必须稳定？
14. 出问题时应该看哪些对象和日志？
```

本篇不重复展开 `05_cudagraph_dispatch_flow.md` 的完整 dispatch 逻辑，也不重复展开 `06_attention_metadata_capture.md` 的 backend metadata 细节；重点放在“并行维度为什么会改变 cudagraph 可用性和 graph key”。

---

## 1. 一句话回答

并行场景下，CUDA graph 不是“单卡模型 forward 的固定 shape replay”。

它还要求：

```text
1. 每个 rank 的本地 tensor shape 可被稳定描述；
2. 所有参与通信的 rank 走相同的 collective 顺序；
3. 需要同步的 rank 对 cudagraph mode / padding / microbatching 达成一致；
4. ForwardContext 中的 attention metadata、DP metadata、slot mapping、ubatch slices 与 capture 时结构一致；
5. 某些通信 kernel / All2All backend 本身必须支持 CUDA graph capture。
```

因此并行机制影响 CUDA graph 的核心不是“能不能并行”，而是：

```text
并行后的计算图和通信图能不能在 capture 与 replay 之间保持同一种形态。
```

---

## 2. 主链路：并行信息在哪里进入 cudagraph dispatch

真实执行入口仍在 `GPUModelRunner.execute_model()`：

```text
_update_states()
  → _prepare_inputs()
  → _determine_batch_execution_and_padding()
  → _get_slot_mappings()
  → _build_attention_metadata()
  → _preprocess()
  → set_forward_context(...)
  → _model_forward()
```

其中并行最集中影响的是：

```text
_determine_batch_execution_and_padding()
set_forward_context()
ForwardContext
worker 层 PP send / recv
UBatchWrapper
DPMetadata
```

`_determine_batch_execution_and_padding()` 的职责包括：

```text
1. 判断 uniform_decode；
2. 处理 encoder / cascade / LoRA 等限制；
3. 先做 sequence parallelism padding；
4. 调 CudagraphDispatcher.dispatch()；
5. 如果 data_parallel_size > 1，再跨 DP rank 协调；
6. 返回 cudagraph_mode / BatchDescriptor / should_ubatch / num_tokens_across_dp。
```

入口位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3810`

核心代码：

```python
num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)

cudagraph_mode, batch_descriptor = dispatch_cudagraph(
    num_tokens_padded,
    disable_full=use_cascade_attn or has_encoder_output,
)

if self.vllm_config.parallel_config.data_parallel_size > 1:
    should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
        coordinate_batch_across_dp(...)
    )
    if num_tokens_across_dp is not None:
        num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded,
            valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
        )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3853` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3905`

最终并行信息进入 `ForwardContext`：

```python
set_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    num_tokens_across_dp=num_tokens_across_dp,
    cudagraph_runtime_mode=cudagraph_mode,
    batch_descriptor=batch_desc,
    ubatch_slices=ubatch_slices_padded,
    slot_mapping=slot_mappings,
    skip_compiled=has_encoder_input,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4303` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4313`

---

## 3. ForwardContext 是并行 + cudagraph 的控制面

`ForwardContext` 中和并行 / CUDA graph 相关的字段：

```python
@dataclass
class ForwardContext:
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
    dp_metadata: DPMetadata | None = None
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None
    ubatch_slices: UBatchSlices | None = None
    skip_compiled: bool = False
```

位置：`code/vllm/vllm/forward_context.py:128` 到 `code/vllm/vllm/forward_context.py:151`

这些字段分别服务于：

| 字段 | 并行含义 | CUDA graph 含义 |
|---|---|---|
| `batch_descriptor` | 当前 rank / DP 协调后的 padded batch 描述 | CUDAGraphWrapper 的 capture/replay key |
| `cudagraph_runtime_mode` | 所有相关 wrapper 走 FULL / PIECEWISE / NONE | 决定 replay、capture 或 pass-through |
| `dp_metadata` | DP / MoE / SP 中跨 rank token 数 | MoE / SP custom op 读取 token 分布 |
| `ubatch_slices` | DBO / microbatch 切分 | UBatchWrapper 按 slice 建立子 context |
| `attn_metadata` | TP/CP/PP/DBO 下的 per-layer attention 描述 | FULL graph 下必须稳定 |
| `slot_mapping` | 每层 KV 写入位置，ubatch 下可能是 list | padding token 必须是安全 slot |
| `skip_compiled` | encoder-decoder 首轮等动态路径 | 可绕过 compiled call |

所以并行机制最终不会作为一堆显式参数传给 `_model_forward()`；它们被折叠进 `ForwardContext`，由 wrapper / attention / MoE custom op 在 forward 内部读取。

---

## 4. BatchDescriptor：CUDA graph key 中包含哪些并行相关信息

`BatchDescriptor` 定义：

```python
@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0
```

位置：`code/vllm/vllm/forward_context.py:29` 到 `code/vllm/vllm/forward_context.py:58`

字段含义：

```text
num_tokens：padded 后 token 数，也是最重要的 graph shape key。
num_reqs：FULL graph 下 attention metadata 可能依赖 request 数；PIECEWISE 可为 None。
uniform：是否所有 request query len 一致；decode / spec decode FULL graph 常依赖它。
has_lora / num_active_loras：LoRA kernel grid 可能依赖 active adapter 数。
```

并行影响主要体现在：

```text
TP / SP：要求 num_tokens 是 TP size 的倍数。
DP：可能把各 rank num_tokens pad 到同一个最大值。
PP：每个 PP rank 的 graph key 本地一致，但输入类型可能不同。
DBO：num_tokens 还会被拆成多个 ubatch。
LoRA：active LoRA 数进入 graph key。
```

`CudagraphDispatcher` 会把真实 token 数映射到 capture size：

```python
num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:132` 到 `code/vllm/vllm/v1/cudagraph_dispatcher.py:156`

---

## 5. Tensor Parallel 对 compile / CUDA graph 的影响

### 5.1 TP 改变每个 rank 上的局部计算 shape

Tensor Parallel 会把权重、head、hidden dim 或 vocab 等维度切到多个 rank。

从 CUDA graph 视角看，TP 影响三类东西：

```text
1. 每个 rank 的 matmul / attention 局部 shape；
2. 每层后的 all-reduce / all-gather / reduce-scatter 通信；
3. 某些 compile pass 是否要融合 GEMM + communication。
```

vLLM 的 cudagraph key 不直接写入 `tensor_parallel_size`，因为 TP size 是进程生命周期内固定配置，不是每轮动态值。

但 TP size 会通过这些路径间接影响 graph：

```text
- 模型加载时每个 rank 的权重 shard 不同；
- compilation hash / pass config 不同；
- sequence parallel padding 要求不同；
- attention backend 的 world size / head 数不同；
- communication custom op / collective kernel 不同。
```

### 5.2 TP collective 顺序必须稳定

CUDA graph replay 捕获的是一段 GPU work，包括通信 kernel 或 NCCL-like op。

所以 TP 下必须保证：

```text
所有 TP rank 在 capture 和 replay 中执行相同顺序的 collectives；
不能某个 rank replay graph，另一个 rank 走 eager 且通信顺序不同；
不能同一层有的 rank进入 fused communication，有的 rank不进入。
```

这也是为什么并行相关的 dispatch / fallback 往往要在 rank 之间协调，而不能每个 rank 完全独立决定。

---

## 6. Sequence Parallel 对 cudagraph size 的影响

### 6.1 SP 会先把 token 数 pad 到 TP size 的倍数

`GPUModelRunner._pad_for_sequence_parallelism()`：

```python
tp_size = self.vllm_config.parallel_config.tensor_parallel_size
if self.compilation_config.pass_config.enable_sp and tp_size > 1:
    return round_up(num_scheduled_tokens, tp_size)
return num_scheduled_tokens
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3407` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3413`

在 `_determine_batch_execution_and_padding()` 中最先调用：

```python
num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3853`

然后如果 SP 开启，会断言：

```python
assert batch_descriptor.num_tokens % tensor_parallel_size == 0
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3869` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3877`

含义：

```text
SP 需要把 token 维在 TP ranks 间切分；
因此 token 数必须能被 TP size 对齐；
CUDA graph capture size 也必须满足这个对齐。
```

### 6.2 spec decode + SP 会进一步约束 capture sizes

`CompilationConfig.adjust_cudagraph_sizes_for_spec_decode()` 中：

```python
multiple_of = uniform_decode_query_len
if tensor_parallel_size > 1 and self.pass_config.enable_sp:
    multiple_of = max(uniform_decode_query_len, tensor_parallel_size)
    if multiple_of % uniform_decode_query_len != 0 or multiple_of % tensor_parallel_size != 0:
        raise ValueError(...)
```

位置：`code/vllm/vllm/config/compilation.py:1462` 到 `code/vllm/vllm/config/compilation.py:1479`

这里的 `uniform_decode_query_len` 通常是：

```text
1 + num_speculative_tokens
```

含义：

```text
spec decode FULL decode graph 要求 capture size 是 num_speculative_tokens + 1 的倍数；
SP 又要求 capture size 是 TP size 的倍数；
两者不能同时满足时，必须调整 num_speculative_tokens、capture sizes 或关闭 SP。
```

### 6.3 SP 与 piecewise cudagraph 有兼容性限制

`CompilationConfig.__post_init__()` 中：

```python
if not self.use_inductor_graph_partition and (enable_sp or fuse_gemm_comms) and self.splitting_ops:
    self.splitting_ops = []
    if self.cudagraph_mode.has_piecewise_cudagraphs():
        self.cudagraph_mode = CUDAGraphMode.FULL
```

位置：`code/vllm/vllm/config/compilation.py:1167` 到 `code/vllm/vllm/config/compilation.py:1184`

日志含义：

```text
Sequence parallelism requires full-graph compilation when use_inductor_graph_partition is off。
Sequence parallelism is incompatible with piecewise cudagraph in this setting。
```

也就是说：

```text
SP / async TP 类 pass 会改变图切分和通信融合；
在非 inductor graph partition 模式下，为了保留 SP，会清空 splitting_ops；
如果原本要求 PIECEWISE cudagraph，会被改成 FULL。
```

---

## 7. Data Parallel 对 CUDA graph 的影响

### 7.1 DP rank 的本地 batch 可能不一样

Data Parallel 下，每个 rank 可以服务不同请求，本地 token 数可能不同：

```text
DP rank 0：num_tokens = 17
DP rank 1：num_tokens = 24
DP rank 2：num_tokens = 3
```

如果每个 rank 独立 dispatch，就可能出现：

```text
rank 0 命中 FULL graph；
rank 1 超出 capture size 走 NONE；
rank 2 pad 到另一个 graph size；
MoE / DP communication 看到不同 token 布局。
```

这会破坏通信顺序或导致 custom op 的跨 rank metadata 不一致。

### 7.2 vLLM 用 all_reduce 协调 DP rank

入口：`coordinate_batch_across_dp()`。

位置：`code/vllm/vllm/v1/worker/dp_utils.py:164`

它会把每个 rank 的信息放进一个 `4 x dp_size` tensor：

```python
tensor_cpu[0][dp_rank] = orig_num_tokens_per_ubatch
tensor_cpu[1][dp_rank] = padded_num_tokens_per_ubatch
tensor_cpu[2][dp_rank] = 1 if should_ubatch else 0
tensor_cpu[3][dp_rank] = cudagraph_mode
```

位置：`code/vllm/vllm/v1/worker/dp_utils.py:36` 到 `code/vllm/vllm/v1/worker/dp_utils.py:54`

然后：

```python
dist.all_reduce(tensor, group=group)
```

位置：`code/vllm/vllm/v1/worker/dp_utils.py:53`

这一步让所有 rank 都知道彼此的：

```text
真实 token 数；
padded token 数；
是否打算 microbatch；
cudagraph runtime mode。
```

### 7.3 DP cudagraph mode 取最小值

```python
return int(tensor[3, :].min().item())
```

位置：`code/vllm/vllm/v1/worker/dp_utils.py:92` 到 `code/vllm/vllm/v1/worker/dp_utils.py:98`

注释说明：

```text
If any rank has NONE (0), all ranks use NONE.
```

含义：

```text
只要有一个 DP rank 不能安全使用 cudagraph，所有 DP rank 都降到相同 mode；
避免某些 rank replay、某些 rank eager 导致通信图不一致。
```

### 7.4 DP padding 只在需要时发生

`_synchronize_dp_ranks()` 中：

```python
should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch
```

位置：`code/vllm/vllm/v1/worker/dp_utils.py:147` 到 `code/vllm/vllm/v1/worker/dp_utils.py:152`

如果需要 DP padding：

```python
max_num_tokens = int(num_tokens_across_dp.max().item())
return torch.tensor([max_num_tokens] * len(num_tokens_across_dp), ...)
```

位置：`code/vllm/vllm/v1/worker/dp_utils.py:77` 到 `code/vllm/vllm/v1/worker/dp_utils.py:89`

含义：

```text
当 cudagraph enabled 或 ubatching enabled 时，DP ranks 会 pad 到相同 token 数；
如果 synced mode 是 NONE 且不 ubatch，可以保留各自真实 token 数。
```

---

## 8. DPMetadata 如何进入 forward

`DPMetadata` 定义：

```python
@dataclass
class DPMetadata:
    num_tokens_across_dp_cpu: torch.Tensor
    local_sizes: list[int] | None = None
```

位置：`code/vllm/vllm/forward_context.py:72` 到 `code/vllm/vllm/forward_context.py:80`

创建逻辑：

```python
assert parallel_config.data_parallel_size > 1
assert parallel_config.is_moe_model is not False
assert num_tokens_across_dp_cpu[dp_rank] == batchsize
return DPMetadata(num_tokens_across_dp_cpu)
```

位置：`code/vllm/vllm/forward_context.py:80` 到 `code/vllm/vllm/forward_context.py:96`

`set_forward_context()` 会接收：

```text
num_tokens_across_dp=num_tokens_across_dp
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4303` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4308`

并在 context 内生成 `dp_metadata`。

这个 metadata 主要给：

```text
MoE / EP custom op；
sequence-parallel MoE；
需要知道跨 DP ranks token 分布的通信 kernel。
```

例如 `DPMetadata.sp_local_sizes()` 会把 DP token 数进一步按 sequence parallel size 展开：

```python
sp_tokens = (num_tokens_across_dp_cpu + sequence_parallel_size - 1) // sequence_parallel_size
sp_tokens = sp_tokens.repeat_interleave(sequence_parallel_size)
```

位置：`code/vllm/vllm/forward_context.py:61` 到 `code/vllm/vllm/forward_context.py:69`

---

## 9. Pipeline Parallel 对 CUDA graph 的影响

### 9.1 不同 PP rank 的 forward 输入不同

Pipeline Parallel 下：

```text
PP first rank：输入 token ids / embeddings / positions。
PP middle rank：输入上一 stage 的 IntermediateTensors。
PP last rank：输出 hidden states，并计算 logits / sampling。
```

非 first rank 在 `GPUWorker.execute_model()` 中先收上一 stage：

```python
tensor_dict, comm_handles, comm_postprocess = get_pp_group().irecv_tensor_dict(...)
intermediate_tensors = AsyncIntermediateTensors(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:853` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:865`

非 last rank 执行完后发送到下一 stage：

```python
self._pp_send_work = get_pp_group().isend_tensor_dict(
    output.tensors,
    all_gather_group=get_tp_group(),
    all_gather_tensors=all_gather_tensors,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:889` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:894`

### 9.2 每个 PP rank 捕获的是本 rank 子模型 graph

由于每个 PP rank 只持有部分 layers，CUDA graph capture 的范围也是本 rank 的 partial forward。

这意味着：

```text
rank 0 graph：token/embedding 输入 → rank0 layers → IntermediateTensors；
中间 rank graph：IntermediateTensors → 本 rank layers → IntermediateTensors；
最后 rank graph：IntermediateTensors → final hidden states。
```

所以 PP 下不能把整个模型的跨 rank pipeline 当成一个 CUDA graph。

vLLM 的做法是：

```text
每个 Worker / ModelRunner 各自决定本地 cudagraph dispatch；
PP 通信由 GPUWorker 在 ModelRunner 前后组织；
通信顺序由 PP group 的 send/recv 保证。
```

### 9.3 PP + SP 时，worker 层会提前计算 batch_desc

在 `GPUWorker.execute_model()` 中，如果：

```text
pipeline_parallel_size > 1
pass_config.enable_sp
forward_pass
```

会提前调用：

```python
_, batch_desc, _, _, _ = self.model_runner._determine_batch_execution_and_padding(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:824` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:846`

然后设置：

```python
all_gather_tensors = {
    "residual": not is_residual_scattered_for_sp(
        self.vllm_config, batch_desc.num_tokens
    )
}
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:847` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:851`

注释也说明这比较“gross”，因为 `_determine_batch_execution_and_padding()` 后面还会在 `execute_model()` 内再调用一次。

含义：

```text
PP 通信是否需要 all-gather 某些 intermediate tensor，取决于 SP 和 padded batch shape；
因此 worker 层在真正 forward 前就需要知道 cudagraph padding 后的 batch_desc。
```

### 9.4 external_launcher + PP 是 V2 unsupported

V2 unsupported features 中：

```python
if distributed_executor_backend == "external_launcher" and pipeline_parallel_size > 1:
    unsupported.append("pipeline parallelism with external_launcher")
```

位置：`code/vllm/vllm/config/vllm.py:2029` 到 `code/vllm/vllm/config/vllm.py:2035`

原因注释：

```text
V2 does not implement the external_launcher PP-output broadcast that V1 uses to keep all ranks in sync。
```

这说明 PP 场景下，rank 间 output / logits 同步也会影响 runner 能力边界。

---

## 10. DBO / microbatching 如何影响 CUDA graph

### 10.1 DBO 先由 DP 协调决定是否所有 rank 都 ubatch

`coordinate_batch_across_dp()` 会检查：

```python
should_attempt_ubatching = check_ubatch_thresholds(
    parallel_config,
    num_tokens_unpadded,
    uniform_decode=uniform_decode,
)
```

位置：`code/vllm/vllm/v1/worker/dp_utils.py:201` 到 `code/vllm/vllm/v1/worker/dp_utils.py:210`

`check_ubatch_thresholds()`：

```python
if not config.use_ubatching:
    return False
if uniform_decode:
    return num_tokens >= config.dbo_decode_token_threshold
else:
    return num_tokens >= config.dbo_prefill_token_threshold
```

位置：`code/vllm/vllm/v1/worker/ubatch_utils.py:38` 到 `code/vllm/vllm/v1/worker/ubatch_utils.py:47`

DP 后处理要求：

```python
should_ubatch = bool(torch.all(tensor[2] == 1).item())
```

位置：`code/vllm/vllm/v1/worker/dp_utils.py:57` 到 `code/vllm/vllm/v1/worker/dp_utils.py:74`

含义：

```text
DBO / ubatching 要么所有 DP rank 都启用，要么都不启用；
不能某些 rank split microbatch，另一些 rank 不 split。
```

### 10.2 ubatch slices 拆分 token / request / metadata

`maybe_create_ubatch_slices()` 会创建两套 slices：

```text
ubatch_slices：按真实 token 范围切分；
ubatch_slices_padded：最后一个 ubatch pad 到 num_tokens_padded / num_reqs_padded。
```

位置：`code/vllm/vllm/v1/worker/ubatch_utils.py:63` 到 `code/vllm/vllm/v1/worker/ubatch_utils.py:114`

其中：

```python
ubatch_slices_padded = _pad_out_ubatch_slices(
    ubatch_slices, num_tokens_padded, num_reqs_padded
)
```

位置：`code/vllm/vllm/v1/worker/ubatch_utils.py:108` 到 `code/vllm/vllm/v1/worker/ubatch_utils.py:114`

### 10.3 UBatchWrapper 为每个 ubatch 创建独立 ForwardContext

`UBatchWrapper._make_ubatch_metadata()` 中：

```python
create_forward_context(
    attn_metadata[i],
    self.vllm_config,
    dp_metadata=dp_metadata[i],
    batch_descriptor=batch_descriptor,
    cudagraph_runtime_mode=cudagraph_runtime_mode,
    slot_mapping=slot_mapping[i],
)
```

位置：`code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:333` 到 `code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:362`

然后每个 ubatch 只拿自己的 input slice：

```python
sliced_input_ids = input_ids[tokens_slice]
sliced_positions = positions[tokens_slice]
sliced_inputs_embeds = inputs_embeds[tokens_slice]
sliced_intermediate_tensors = intermediate_tensors[tokens_slice]
```

位置：`code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:400` 到 `code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:429`

含义：

```text
DBO 下 ForwardContext 不再是单个 batch 的 context；
外层 context 中 attn_metadata / slot_mapping 是 list；
每个 ubatch 线程内部再切换到自己的 context。
```

### 10.4 ubatch CUDA graph 是按总 token 数缓存

`UBatchWrapper.__call__()` 中：

```python
num_tokens = sum(ubatch_slice.num_tokens for ubatch_slice in ubatch_slices)
```

位置：`code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:456` 到 `code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:459`

如果 FULL 且没有 graph：

```python
self._capture_ubatches(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:483` 到 `code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:501`

如果已有 graph：

```python
cudagraph_metadata.cudagraph.replay()
return cudagraph_metadata.outputs
```

位置：`code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:503` 到 `code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:511`

所以 DBO 的 CUDA graph 形态不是简单的单 forward replay，而是：

```text
主线程 + ubatch 线程 + compute stream / comm stream + per-ubatch ForwardContext。
```

### 10.5 V2 当前不支持 DBO

V2 unsupported features 中：

```python
if self.parallel_config.enable_dbo:
    unsupported.append("dual batch overlap")
```

位置：`code/vllm/vllm/config/vllm.py:2061` 到 `code/vllm/vllm/config/vllm.py:2062`

---

## 11. Expert Parallel / MoE 对 CUDA graph 的影响

### 11.1 MoE 需要 DP token 分布

`ParallelConfig` 中：

```text
data_parallel_size：MoE layers will be sharded according to TP x DP。
enable_expert_parallel：Use expert parallelism instead of tensor parallelism for MoE layers。
all2all_backend：MoE expert parallel communication backend。
```

位置：`code/vllm/vllm/config/parallel.py:126` 到 `code/vllm/vllm/config/parallel.py:195`

MoE / EP 的关键问题是：

```text
每个 rank 的 token 会根据 routing 被发给不同 expert；
All2All / allgather / reducescatter 的通信量依赖 token 分布；
CUDA graph replay 要求这种通信 kernel 的 launch 形态和 buffer 结构稳定。
```

因此 vLLM 需要：

```text
DPMetadata.num_tokens_across_dp_cpu
sequence parallel local_sizes
MoE layer names in ForwardContext
stable custom op execution order
```

### 11.2 DeepEP high-throughput 会禁用 CUDA graphs

`CompilationConfig.__post_init__()` 中：

```python
if all2all_backend == "deepep_high_throughput" and data_parallel_size > 1 and cudagraph_mode != NONE:
    self.cudagraph_mode = CUDAGraphMode.NONE
```

位置：`code/vllm/vllm/config/compilation.py:1186` 到 `code/vllm/vllm/config/compilation.py:1202`

日志说明：

```text
DeepEP high-throughput kernels are optimized for prefill and are incompatible with CUDA Graphs。
```

所以 EP/MoE backend 不是只影响性能，还可能直接改变 `cudagraph_mode`。

### 11.3 sequence-parallel MoE 的触发条件

`ParallelConfig.use_sequence_parallel_moe`：

```python
return (
    all2all_backend in (...)
    and enable_expert_parallel
    and tensor_parallel_size > 1
    and data_parallel_size > 1
)
```

位置：`code/vllm/vllm/config/parallel.py:625` 到 `code/vllm/vllm/config/parallel.py:648`

注释解释：

```text
attention o_proj 后 all_reduce 让 TP group 输入被复制；
如果 EP + DeepEP All2All 下继续复制 token，会造成重复计算和通信；
因此要用 sequence parallel 避免多余工作。
```

这进一步说明：

```text
TP、DP、EP 不是彼此独立影响 cudagraph；
MoE 通信形态可能反过来要求 SP；
SP 又要求 token padding 和 graph size 对齐。
```

---

## 12. Context Parallel / DCP / PCP 的影响

### 12.1 ParallelConfig 约束 DCP 必须整除 TP

`ParallelConfig` 中：

```python
if tensor_parallel_size % decode_context_parallel_size != 0:
    raise ValueError(...)
```

位置：`code/vllm/vllm/config/parallel.py:490` 到 `code/vllm/vllm/config/parallel.py:499`

DCP 复用 TP group，不改变总 world size，但会把 TP group 再切分成 context parallel groups。

### 12.2 attention backend 必须支持 DCP / PCP 能力

`check_attention_cp_compatibility()`：

```python
if dcp_size > 1:
    assert layer_impl.need_to_return_lse_for_decode

if pcp_size > 1:
    assert layer_impl.supports_pcp
```

位置：`code/vllm/vllm/v1/worker/cp_utils.py:14` 到 `code/vllm/vllm/v1/worker/cp_utils.py:44`

如果 spec decode + non-trivial interleave：

```python
if speculative_config is not None and interleave_size > 1:
    assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size
```

位置：`code/vllm/vllm/v1/worker/cp_utils.py:25` 到 `code/vllm/vllm/v1/worker/cp_utils.py:29`

这说明：

```text
CP 不只是改变 block table；
attention backend 必须能返回 decode LSE、支持 PCP、支持 spec/MTP 下的 interleave。
```

### 12.3 V2 当前不支持 PCP

V2 unsupported features 中：

```python
if self.parallel_config.prefill_context_parallel_size > 1:
    unsupported.append("prefill context parallelism")
```

位置：`code/vllm/vllm/config/vllm.py:2017` 到 `code/vllm/vllm/config/vllm.py:2018`

---

## 13. LoRA 与并行 graph key

LoRA 不一定是并行机制，但它会和并行 batch 一起影响 CUDA graph key。

`BatchDescriptor` 中有：

```text
has_lora
num_active_loras
```

位置：`code/vllm/vllm/forward_context.py:47` 到 `code/vllm/vllm/forward_context.py:58`

`CudagraphDispatcher._get_lora_cases()` 会根据配置生成 capture cases：

```python
if cudagraph_specialize_lora:
    return [0] + captured_counts
else:
    return [max_loras + 1]
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:111` 到 `code/vllm/vllm/v1/cudagraph_dispatcher.py:130`

运行时如果 active LoRA 数不是直接 capture 的值，会映射到最小的足够 bucket：

```python
idx = bisect.bisect_left(self.captured_lora_counts, num_active_loras)
effective_num_active_loras = self.captured_lora_counts[idx]
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:283` 到 `code/vllm/vllm/v1/cudagraph_dispatcher.py:300`

含义：

```text
某些 LoRA kernel grid 依赖 active LoRA 数；
graph key 必须区分 no-LoRA / with-LoRA / active count bucket；
否则 replay 的 kernel launch 形态可能不匹配。
```

---

## 14. V2 ModelRunner 中并行 + compile/cudagraph 的 unsupported

`VllmConfig._get_v2_model_runner_unsupported_features()` 明确列出一些并行 / compile 限制：

```text
prefill context parallelism
stock torch.compile
sequence parallelism
pipeline parallelism with external_launcher
dual batch overlap
elastic expert parallelism
routed experts capture
```

位置：`code/vllm/vllm/config/vllm.py:2017` 到 `code/vllm/vllm/config/vllm.py:2069`

这说明：

```text
V1 GPUModelRunner 中已有的并行 + cudagraph 处理，不一定已经迁移到 V2 runner；
尤其是 PP output broadcast、DBO、SP、PCP、routed experts capture 等路径。
```

---

## 15. capture 阶段并行信息如何固定

CUDA graph capture 在 warmup / dummy run 阶段发生。

并行场景下 capture 不是只固定 `num_tokens`，还要固定：

```text
TP / SP：token 数是否能被 TP size 切分；
DP：是否需要 padding 到跨 rank 最大 token 数；
PP：本 rank 是 first/middle/last，输入是否是 intermediate_tensors；
DBO：是否按 ubatch 切分，每个 ubatch 的 token/request slice；
EP/MoE：All2All backend 和 DP token 分布表达；
CP：attention metadata 中 local seq lens / LSE / interleave 结构。
```

真正进入 wrapper 的 key 仍是：

```text
CUDAGraphMode + BatchDescriptor
```

但这个 key 的生成已经吸收了并行约束。

---

## 16. replay 阶段哪些内容必须一致

### 16.1 必须一致的内容

```text
1. cudagraph_runtime_mode：FULL / PIECEWISE / NONE；
2. batch_descriptor：num_tokens / num_reqs / uniform / LoRA case；
3. tensor 地址：input buffers / metadata buffers / graph input tensors；
4. collective 顺序：TP / PP / DP / EP 通信不能错序；
5. attention metadata 结构：backend metadata 类型和关键 shape；
6. ubatch slice 数和线程/stream 调度结构；
7. DP ranks 是否都使用同一种 mode 和 padding策略。
```

### 16.2 可以变的内容

```text
1. block_table tensor 内容；
2. slot_mapping tensor 内容；
3. seq_lens / query_start_loc 内容；
4. token ids / positions 内容；
5. DPMetadata 中每个 rank 的 token 数内容；
6. MoE routing 结果内容。
```

前提是：

```text
tensor shape 和地址稳定；
通信 buffer 形态稳定；
padding 区域被安全填充；
backend 允许这些内容在 replay 前更新。
```

---

## 17. 典型组合场景

### 17.1 TP only + 普通 decode + FULL cudagraph

```text
TP size > 1
DP = PP = 1
uniform_decode = True
num_tokens = num_reqs
```

特点：

```text
每个 TP rank capture 自己的 shard forward；
通信 collectives 在每个 rank 上顺序稳定；
BatchDescriptor 通常按 num_tokens / num_reqs / uniform 匹配 FULL key。
```

### 17.2 TP + SP + spec decode

```text
enable_sp=True
uniform_decode_query_len = 1 + num_speculative_tokens
```

要求：

```text
capture size 既是 uniform_decode_query_len 的倍数，
又满足 TP size / SP 切分要求。
```

如果不能满足，会在 `adjust_cudagraph_sizes_for_spec_decode()` 中报错或要求调整配置。

### 17.3 DP + MoE + CUDA graph

```text
data_parallel_size > 1
is_moe_model=True
cudagraph_mode != NONE
```

流程：

```text
本地 dispatch → DP all_reduce 协调 → mode 取 min → 必要时各 rank pad 到最大 token 数 → 重新 dispatch → ForwardContext 写入 DPMetadata。
```

核心原因：

```text
MoE / EP communication 需要跨 DP ranks 一致的 token 分布描述和 graph mode。
```

### 17.4 DP 中某个 rank 不能用 cudagraph

```text
rank0：FULL
rank1：NONE
rank2：FULL
```

协调后：

```text
synced_cudagraph_mode = min(...) = NONE
所有 rank 都走 NONE
```

这样保证 collective 顺序一致。

### 17.5 PP + SP

```text
pipeline_parallel_size > 1
enable_sp=True
```

worker 层会提前调用 `_determine_batch_execution_and_padding()` 得到 `batch_desc`，用来决定 PP tensor send/recv 时的 `all_gather_tensors`。

这说明：

```text
PP 通信形态依赖 cudagraph / SP padding 后的 batch shape。
```

### 17.6 DBO + FULL graph

```text
enable_dbo=True 或 ubatch_size > 1
should_ubatch=True
cudagraph_runtime_mode=FULL
```

执行形态：

```text
UBatchWrapper
  → 为每个 ubatch 创建 ForwardContext
  → 多线程 / 多 stream 运行
  → 首次按 num_tokens capture ubatch graph
  → 后续 replay 整个 ubatch 调度 graph
```

### 17.7 DeepEP high-throughput + DP

```text
all2all_backend = deepep_high_throughput
data_parallel_size > 1
```

配置阶段直接：

```text
cudagraph_mode = NONE
```

因为该 All2All backend 当前与 CUDA Graph 不兼容。

---

## 18. 常见误解

### 18.1 TP size 会出现在 BatchDescriptor 里吗？

不会直接出现。

TP size 是静态配置，影响模型 shard、compile pass、通信 op 和 SP padding；而不是每轮动态 graph key。

### 18.2 DP rank 可以各自决定 cudagraph mode 吗？

不能完全各自决定。

本地会先 dispatch，但随后通过 DP all_reduce 同步，最终 mode 取所有 rank 的最小值。

### 18.3 cudagraph fallback 到 NONE 是否等于不 compile？

不等于。

`CUDAGraphMode.NONE` 只表示不 replay CUDA graph；底层仍可能是 torch.compile / vLLM compile 后的 callable。

### 18.4 PP 下能 capture 跨 rank pipeline 吗？

不能按单个 CUDA graph capture 整个 pipeline。

每个 PP rank capture 自己本地 forward；rank 间 send/recv 由 Worker 层组织。

### 18.5 DBO 是简单把 batch 拆两半吗？

不是。

它还要拆：

```text
input_ids / positions / inputs_embeds / intermediate_tensors；
attention metadata；
slot_mapping；
DPMetadata；
ForwardContext；
compute / comm stream 调度。
```

### 18.6 piecewise cudagraph 一定更兼容 SP 吗？

不一定。

在 `use_inductor_graph_partition=False` 且启用 SP / async TP 时，vLLM 会清空 `splitting_ops`，并把 piecewise cudagraph 模式改成 FULL。

---

## 19. 排查并行 + cudagraph 问题看哪些对象

优先看：

```text
ParallelConfig：
  tensor_parallel_size
  pipeline_parallel_size
  data_parallel_size
  enable_expert_parallel
  all2all_backend
  enable_dbo / ubatch_size
  decode_context_parallel_size / prefill_context_parallel_size

CompilationConfig：
  mode
  cudagraph_mode
  cudagraph_capture_sizes
  max_cudagraph_capture_size
  pass_config.enable_sp
  pass_config.fuse_gemm_comms
  splitting_ops
  use_inductor_graph_partition

GPUModelRunner runtime：
  cudagraph_mode
  batch_descriptor
  should_ubatch
  num_tokens_across_dp
  ubatch_slices_padded
  cudagraph_stats

ForwardContext：
  dp_metadata
  batch_descriptor
  cudagraph_runtime_mode
  ubatch_slices
  attn_metadata
  slot_mapping

DP coordination：
  orig_num_tokens_per_ubatch
  padded_num_tokens_per_ubatch
  synced_cudagraph_mode
  should_ubatch
```

如果开启：

```text
observability_config.cudagraph_metrics=True
```

`GPUModelRunner` 会生成：

```python
CUDAGraphStat(
    num_unpadded_tokens=num_tokens,
    num_padded_tokens=batch_descriptor.num_tokens,
    num_paddings=batch_descriptor.num_tokens - num_tokens,
    runtime_mode=str(cudagraph_mode),
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3907` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3914`

这对判断“为什么 pad 了这么多 / 为什么 fallback 到 NONE”很有用。

---

## 20. 从“回答问题”的角度总结

如果要问：

```text
并行机制如何影响 CUDA graph / compile？
```

可以回答：

```text
并行机制会把 CUDA graph 的约束从“单个 rank 的 tensor shape 固定”扩展为“所有相关 rank 的计算 shape、通信顺序、metadata 结构和 runtime mode 都固定”。

Tensor Parallel 改变每个 rank 的局部计算图，并引入 all-reduce / all-gather / reduce-scatter 等 collective；如果启用 Sequence Parallel，num_tokens 还必须按 TP size 对齐，spec decode 下 cudagraph capture sizes 还要同时满足 num_speculative_tokens + 1 和 TP size 的倍数要求。

Pipeline Parallel 让不同 PP rank 的 forward 输入/输出不同：首 rank 处理 token ids，中间 rank 处理 IntermediateTensors，末 rank 产出 final hidden states；每个 rank capture 的只是本地 stage graph，PP send/recv 在 Worker 层组织。PP + SP 时，worker 层甚至需要提前根据 padded BatchDescriptor 决定 intermediate tensor 是否 all-gather。

Data Parallel 下各 rank 本地 batch 可能不同，所以 vLLM 会用 all_reduce 同步每个 rank 的真实 token 数、padded token 数、是否 ubatch、cudagraph mode。最终 cudagraph mode 取最小值：只要某个 rank 是 NONE，所有 rank 都降到 NONE；如果启用 cudagraph 或 ubatching，还会把各 DP rank pad 到相同 token 数，并通过 DPMetadata 把跨 rank token 分布放进 ForwardContext。

DBO / microbatching 会把一个 batch 拆成多个 ubatch，并为每个 ubatch 创建独立 ForwardContext、attention metadata、slot_mapping 和 DPMetadata；FULL cudagraph 下 UBatchWrapper 还会 capture / replay 整个 ubatch 调度过程。

Expert Parallel / MoE 会引入 All2All、routing、DP token 分布和 sequence-parallel MoE 等约束；某些 backend 例如 DeepEP high-throughput 当前会直接禁用 CUDA Graph。Context Parallel / DCP / PCP 则要求 attention backend 支持对应的 LSE、PCP、interleave 能力。

所以并行场景下 cudagraph 的难点不是某个 rank 能不能 replay，而是所有 rank 能不能用一致的 shape、metadata 和通信图 replay；不能满足时，vLLM 会通过 padding、mode 同步、禁用 FULL、改用 PIECEWISE 或 fallback 到 NONE 来保持正确性。
```

---

## 21. 最关键流程图

```text
单 rank 本地 dispatch

num_tokens / num_reqs / uniform_decode
  → _pad_for_sequence_parallelism()
  → CudagraphDispatcher.dispatch()
  → local cudagraph_mode + BatchDescriptor
```

```text
DP 协调

local num_tokens / padded_tokens / should_ubatch / cudagraph_mode
  → all_reduce across DP group
  → synced_cudagraph_mode = min(all modes)
  → if cudagraph or ubatch:
       pad all ranks to max padded tokens
  → re-dispatch with valid_modes={synced_cudagraph_mode}
  → num_tokens_across_dp
```

```text
ForwardContext

BatchDescriptor
  + cudagraph_runtime_mode
  + DPMetadata(num_tokens_across_dp)
  + ubatch_slices
  + attn_metadata
  + slot_mapping
  → self.model(...)
      → CUDAGraphWrapper / UBatchWrapper / attention / MoE custom ops
```

```text
PP 外层

if not first PP rank:
  recv IntermediateTensors

ModelRunner.execute_model()
  → local cudagraph / compile / attention

if not last PP rank:
  send IntermediateTensors
else:
  compute logits / sample
```

---

## 22. 最小心智模型

如果只记一个模型：

```text
CUDA graph key 描述的是“本 rank 要 replay 的本地计算形态”；
并行机制决定这个本地形态能否和其他 rank 的通信形态同时稳定。
```

再压缩成一句话：

```text
并行场景下，vLLM 必须同时固定计算 shape 和通信 shape；TP/SP 改 token 对齐，PP 改输入输出边界，DP/DBO 要跨 rank 同步 mode 和 padding，EP/CP 要 backend 支持对应通信与 metadata，任何一环不稳定就退到更保守的 cudagraph mode 或 eager 路径。
```
