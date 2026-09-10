# 12. 算子问题如何定位和分析？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\platforms\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\logger.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\selector.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backend.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\quantization\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\fused_moe\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\profiler\wrapper.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\profiler\layerwise_profile.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\triton_utils\jit_monitor.py`

这个问题关注：当出现 fallback、kernel 不支持、shape mismatch、NaN、CUDA Graph capture 失败、性能异常时，如何判断问题属于哪个算子族、实际走了哪个 backend、该从哪些日志和 profiler 入口定位。

---

## 1. 一句话回答

算子 debug 的核心是先确认“实际走了哪个 backend”，再确认“输入 shape / dtype / layout / metadata 是否满足这个 backend 的要求”，最后用 profiler 把 Python 层模块、torch op、CUDA kernel 和通信 kernel 对上。

最小排查链路是：

```text
现象
  → 定位算子族
  → 确认实际 backend
  → 检查 shape / dtype / layout / metadata
  → 观察 profiler kernel / record_function range
  → 判断是配置、fallback、实现 bug、硬件限制还是动态图问题
```

不要一开始就假设是 CUDA kernel bug。很多问题实际来自：

```text
backend 没选中；
shape / dtype 不满足 backend；
quantized weight metadata 不匹配；
attention metadata 和 KV cache layout 不一致；
CUDA Graph runtime 降级；
输出侧 logprobs / sampling 同步开销过高。
```

---

## 2. 第一步：先按现象定位算子族

可以先按报错位置和现象分组。

```text
attention / KV cache：
  报错里有 attention backend、slot_mapping、block_table、kv_cache、paged_attention、flash_attn、flashinfer、triton_attn。

linear / quantization：
  报错里有 qweight、scales、zero_points、pack_factor、marlin、cutlass、scaled_mm、gptq、awq、fp8。

MoE：
  报错里有 fused_moe、topk_ids、topk_weights、expert、all2all、dispatch、combine、router_logits。

sampling / logits：
  报错里有 sampler、top_k、top_p、logprobs、grammar bitmask、bad_words、rejection_sampler。

parallel communication：
  报错里有 all_reduce、all_gather、reduce_scatter、send_tensor_dict、recv_tensor_dict、all2all、NCCL。

CUDA Graph / compile：
  报错里有 cudagraph、capture、replay、graph break、torch.compile、dynamo、inductor、compile_sizes。
```

定位算子族之后，再看对应 backend 和 metadata。

---

## 3. 第二步：确认实际 backend

### 3.1 attention backend

attention backend 选择入口：

```python
get_attn_backend(...)
```

位置：`vllm/v1/attention/selector.py:54`

CUDA 平台会打印类似日志：

```text
Using X attention backend out of potential backends: [...]
Some attention backends are not valid ... Reasons: {...}
Using Y KV cache layout for X backend.
```

相关位置：

```text
vllm/platforms/cuda.py:395
vllm/platforms/cuda.py:436
vllm/v1/attention/selector.py:138
```

如果用户显式指定 backend，验证失败会直接报错：

```text
Selected backend X is not valid for this configuration. Reason: [...]
```

位置：`vllm/platforms/cuda.py:360`

### 3.2 sampling backend

sampling backend 在 `TopKTopPSampler` 初始化和 forward 中决定。

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:70`

看日志：

```text
Using FlashInfer for top-p & top-k sampling.
FlashInfer top-p/top-k sampling unavailable ... falling back.
aiter sampler ... falling back.
xpu kernel topk_topp_sampler ... falling back.
```

还要注意运行时 fallback：

```text
per-request generators → FlashInfer / XPU / aiter 可能 fallback；
processed logits/logprobs mode → FlashInfer sampler 不可用；
use_fp64_gumbel=True → fallback native；
k 和 p 都为空 → native。
```

### 3.3 quantization backend

量化 config 入口：

```python
get_quantization_config(quantization)
```

位置：`vllm/model_executor/layers/quantization/__init__.py:107`

排查时确认：

```text
checkpoint quant_method 是什么；
CLI / config 里的 quantization 是什么；
实际 QuantizationConfig class 是什么；
每个 linear layer 的 quant_method 是什么；
profiler 里是否出现 marlin / cutlass / scaled_mm / triton / torch mm。
```

### 3.4 MoE backend

MoE 需要同时确认：

```text
router / grouped_topk backend；
fused_moe backend；
expert GEMM backend；
EP all2all backend；
quantized expert weight backend。
```

相关源码：

```text
vllm/model_executor/layers/fused_moe/
vllm/distributed/device_communicators/all2all.py
vllm/distributed/parallel_state.py:1186
```

---

## 4. 第三步：检查 shape / dtype / layout / metadata

不同算子族的关键检查项不同。

### 4.1 attention

重点看：

```text
head_size
num_heads / num_kv_heads
block_size
kv_cache_dtype
KV cache layout
slot_mapping shape
block_table shape
query_len / max_query_len
prefill / decode metadata
cascade attention metadata
CP / DCP 是否要求 LSE
```

backend 会通过 `validate_configuration()` 检查部分条件。

定义入口：`vllm/v1/attention/backend.py:276`

平台层调用：`vllm/platforms/cuda.py:337`

### 4.2 linear / quantization

重点看：

```text
input_size_per_partition
output_partition_sizes
TP rank / TP size
qweight shape
scale / zero point shape
packed_dim / packed_factor
group_size / block_size
Marlin tile offset
FP8 block scale shape
bias 是否重复添加
```

TP linear 的 weight loader 会根据 shard 维度 narrow 权重：

```text
ColumnParallelLinear：按 output_dim shard；
RowParallelLinear：按 input_dim shard；
MergedColumnParallelLinear：按 fused shard id 和 packed layout shard。
```

位置：`vllm/model_executor/layers/linear.py`

### 4.3 vocab / logits

重点看：

```text
vocab_size 是否 padded；
org vocab 和 added vocab 是否分开 shard；
TP gather 后是否按 sharded_to_full_mapping reindex；
logits_indices 是否正确；
padding vocab 是否被 mask。
```

源码位置：`vllm/model_executor/layers/vocab_parallel_embedding.py`

### 4.4 MoE

重点看：

```text
topk_ids 是否越界；
topk_weights dtype / shape；
expert map 是否和 EP group 一致；
每个 expert token count 是否合理；
all2all dispatch 后 token buffer 是否按 expert 排列；
combine 后是否恢复原 token 顺序；
quantized expert weight shape 是否匹配。
```

### 4.5 sampling / logprobs

重点看：

```text
logits shape: [batch, vocab]
temperature / top_k / top_p shape: [batch]
allowed_token_ids_mask shape
bad_words 与 output_token_ids 是否同步
logprobs_mode 是否导致 backend fallback
max_num_logprobs 是否过大
spec decode target_logits_indices / bonus_logits_indices 是否对齐
```

---

## 5. 日志入口怎么看

vLLM 里很多 backend 选择和 fallback 使用：

```python
logger.info_once(...)
logger.warning_once(...)
logger.debug_once(...)
```

这些日志是排查 backend 的第一入口。

重点关注：

```text
Using ... attention backend
Using ... KV cache layout
Some attention backends are not valid ... Reasons
FlashInfer sampler unavailable ... falling back
aiter sampler ... falling back
--block-size precluded higher-priority backend(s)
Profiler stopped successfully
Max profiling iterations reached
```

如果没有看到 debug 级别的 backend invalid reasons，可以提高日志级别后再运行最小复现。

---

## 6. profiler 的入口

vLLM profiler wrapper 在：`vllm/profiler/wrapper.py`

核心类：

```text
WorkerProfiler
TorchProfilerWrapper
CudaProfilerWrapper
```

`TorchProfilerWrapper` 用的是：

```python
torch.profiler.profile(
    activities=[...],
    schedule=...,
    record_shapes=...,
    profile_memory=...,
    with_stack=...,
    with_flops=...,
    on_trace_ready=...,
)
```

位置：`vllm/profiler/wrapper.py:218`

它会把 trace 写到：

```text
profiler_config.torch_profiler_dir
```

位置：`vllm/profiler/wrapper.py:172`

如果启用 dump CUDA time table，会输出按 `self_cuda_time_total` 排序的表：

```python
self.profiler.key_averages().table(sort_by="self_cuda_time_total")
```

位置：`vllm/profiler/wrapper.py:269`

---

## 7. start_profile / stop_profile 链路

服务 API 入口：

```text
POST /start_profile
POST /stop_profile
```

位置：`vllm/entrypoints/serve/profile/api_router.py:21`

LLM / Engine 入口：

```text
LLM.start_profile()
LLM.stop_profile()
LLMEngine.start_profile()
LLMEngine.stop_profile()
AsyncLLM.start_profile()
AsyncLLM.stop_profile()
```

相关位置：

```text
vllm/entrypoints/llm.py:787
vllm/v1/engine/llm_engine.py:336
vllm/v1/engine/async_llm.py:905
```

profiler 支持：

```text
delay_iterations：start_profile 后延迟多少 step 开始；
max_iterations：最多记录多少 worker step；
wait / warmup / active schedule；
record_shapes / memory / stack / flops。
```

位置：`vllm/profiler/wrapper.py:19`

---

## 8. record_function range 如何对齐源码阶段

V1 里很多关键阶段用：

```python
record_function_or_nullcontext("...")
```

位置：`vllm/v1/utils.py:733`

GPUModelRunner 中常见 range：

```text
gpu_model_runner: preprocess
gpu_model_runner: forward
gpu_model_runner: postprocess
gpu_model_runner: sample
gpu_model_runner: draft
gpu_model_runner: bookkeep
gpu_model_runner: eplb
gpu_model_runner: ModelRunnerOutput
```

位置：`vllm/v1/worker/gpu_model_runner.py:4082` 起

Scheduler / Engine 也有 range：

```text
schedule: allocate_slots
schedule: update_after_schedule
llm_engine step: get_output
llm_engine step: process_outputs
```

相关位置：

```text
vllm/v1/core/sched/scheduler.py
vllm/v1/engine/llm_engine.py
```

排查时可以先看耗时落在哪个 range：

```text
forward 慢：看 attention / linear / MoE / communication；
sample 慢：看 top-k/top-p、logprobs、grammar、rejection sampler；
bookkeep 慢：看 D2H copy、logprobs tensor 转 list、prompt logprobs；
preprocess 慢：看 input preparation、MM encoder、slot mapping；
schedule 慢：看 KV allocation、prefix cache、request 状态。
```

---

## 9. layerwise profiling

层级 profiling 入口在：`vllm/profiler/layerwise_profile.py`

`LayerwiseProfileResults` 会把 profiler event 转成：

```text
model_stats_tree：按模型模块层级统计 CPU / CUDA time；
summary_stats_tree：按 kernel / torch op 汇总 CUDA time。
```

位置：`vllm/profiler/layerwise_profile.py:82`

可输出：

```python
print_model_table()
print_summary_table()
export_model_stats_table_csv(...)
export_summary_stats_table_csv(...)
convert_stats_to_dict()
```

位置：`vllm/profiler/layerwise_profile.py:99`

适合回答：

```text
到底是哪一层慢？
是 attention 慢还是 MLP 慢？
是 torch op 慢还是 CUDA kernel 慢？
某个 module 下实际调用了哪些底层 op？
```

---

## 10. 如何识别 torch fallback

torch fallback 通常有几种表现：

```text
日志里出现 falling back / unavailable / unsupported；
profiler 中 kernel name 不再是 flash_attn / flashinfer / marlin / cutlass / fused_moe；
出现 aten::matmul、aten::mm、aten::sort、aten::topk、aten::softmax、aten::scatter 等 torch op；
kernel launch 数量明显增加；
CPU time 或 CUDA sync 增多；
throughput 明显下降但结果仍正确。
```

典型例子：

```text
sampling top-p fallback 到 PyTorch sort；
FlashInfer sampler 因 per-request generators fallback；
quantized linear 没走 Marlin / CUTLASS；
attention backend 从 FlashAttention fallback 到 Triton / Flex / torch SDPA；
MoE 没走 fused kernel。
```

---

## 11. NaN / Inf 如何排查

V1 GPUModelRunner 有专门统计 logits NaN 的路径：

```python
if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
    num_nans_in_logits = self._get_nans_in_logits(logits)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3617`

结果会进入 `ModelRunnerOutput`：

```text
num_nans_in_logits: dict[str, int] | None
```

相关位置：`vllm/v1/outputs.py`

排查 NaN 时按顺序看：

```text
1. NaN 是否只在 logits 出现，还是 hidden states 已经 NaN；
2. 是否只在特定 dtype / quantization 下出现；
3. 是否只在某个 attention backend 下出现；
4. 是否只在长上下文、特定 RoPE scaling、FP8 KV cache 下出现；
5. 是否与 temperature、penalty、logits processor 有关；
6. 是否是 vocab padding / invalid token 导致的 -inf 传播问题；
7. 是否只有 spec decode / rejection sampler 路径出现。
```

常用切分方法：

```text
先禁用 sampling 复杂功能：logprobs、grammar、spec decode；
再切换 attention backend；
再切换 dtype / quantization；
最后缩短 prompt 和 batch 构造最小复现。
```

---

## 12. shape mismatch 如何定位

shape mismatch 一般要同时看“全局 shape”和“rank-local shape”。

### 12.1 TP linear

检查：

```text
input_size / output_size
input_size_per_partition / output_size_per_partition
TP size 是否整除
weight shard narrow 的 dim 是否正确
quantized packed dim 是否按 packed_factor 调整
merged shard id 是否正确
```

对应源码：`vllm/model_executor/layers/linear.py`

### 12.2 attention

检查：

```text
num_tokens / num_tokens_padded
num_reqs / num_reqs_padded
max_query_len
slot_mapping
block table
KV cache shape
backend required layout
```

`GPUModelRunner` 会在 full cudagraph 下用 padded shape 构造 slot mapping / attention metadata。

位置：`vllm/v1/worker/gpu_model_runner.py:4244`

### 12.3 PP

检查：

```text
IntermediateTensors keys；
上一 stage 输出和下一 stage 预期是否一致；
send_tensor_dict / recv_tensor_dict metadata；
TP all_gather optimization 是否误用于非 replicated tensor。
```

位置：`vllm/distributed/parallel_state.py:1021`

### 12.4 MoE

检查：

```text
topk_ids shape；
hidden_states 与 topk_weights batch token 数；
expert id 是否属于本 EP group；
dispatch 后 token 数是否与 combine 匹配。
```

---

## 13. CUDA Graph capture / replay 问题

CUDA Graph 问题先看 runtime mode。

如果启用 cudagraph metrics，会生成：

```python
CUDAGraphStat(
    num_unpadded_tokens=num_tokens,
    num_padded_tokens=batch_descriptor.num_tokens,
    num_paddings=batch_descriptor.num_tokens - num_tokens,
    runtime_mode=str(cudagraph_mode),
)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3907`

排查顺序：

```text
1. cudagraph_mode 是否实际为 FULL / PIECEWISE，还是 NONE；
2. num_tokens 是否超过 max_cudagraph_capture_size；
3. batch descriptor 是否命中 capture key；
4. DP rank 是否把 mode 同步降级为 NONE；
5. 是否因 cascade attention / encoder input / calculate_kv_scales 禁用；
6. LoRA active count 是否没有匹配 captured graph；
7. attention backend 是否支持当前 cudagraph mode；
8. compile_sizes 是否会被 cudagraph padding 改变。
```

相关源码：

```text
vllm/v1/cudagraph_dispatcher.py
vllm/v1/worker/gpu_model_runner.py:3810
vllm/v1/worker/gpu_model_runner.py:6584
```

---

## 14. torch compile 问题

torch compile 问题常见表现：

```text
首次请求特别慢；
batch size 变化时反复 recompile；
Dynamo graph break；
某个 custom op 不可 trace；
NVTX hook / Python side control flow 破坏 fullgraph；
compile size 与 runtime padding 不匹配。
```

V1 中 STOCK_TORCH_COMPILE 会直接：

```python
self.model.compile(fullgraph=True, backend=backend)
```

位置：`vllm/v1/worker/gpu_model_runner.py:5273`

而 layerwise NVTX hooks 在 STOCK_TORCH_COMPILE 下会跳过，因为 hook function 会被 Dynamo trace：

位置：`vllm/v1/worker/gpu_model_runner.py:3941`

排查方法：

```text
先区分是 torch compile 问题还是 CUDA Graph replay 问题；
固定 batch size 复现；
关闭复杂功能如 LoRA / spec decode / grammar；
查看 graph break 日志；
检查 compile_sizes 与 capture_sizes。
```

---

## 15. communication kernel 如何排查

通信问题通常表现为：

```text
NCCL hang；
某个 rank shape 不一致；
all_reduce / all_gather 耗时异常；
PP send / recv metadata 不匹配；
EP all2all dispatch/combine 慢；
DP padding 让 token 数被放大。
```

定位方向：

```text
TP：看 tensor_model_parallel_all_reduce / all_gather；
PP：看 send_tensor_dict / recv_tensor_dict；
DP：看 coordinate_batch_across_dp；
EP：看 GroupCoordinator.dispatch / combine 和 all2all communicator；
CP：看 attention backend 是否需要 LSE merge。
```

对应源码：

```text
vllm/distributed/communication_op.py
vllm/distributed/parallel_state.py
vllm/distributed/device_communicators/all2all.py
vllm/v1/worker/dp_utils.py
vllm/v1/worker/cp_utils.py
```

profiler 中重点看：

```text
ncclAllReduce
ncclAllGather
ncclReduceScatter
ncclSend / ncclRecv
all2all dispatch / combine kernel
CPU wait 或 stream synchronization
```

---

## 16. 最小复现 batch 怎么构造

最小复现不是只缩短 prompt，而是逐步固定变量。

建议顺序：

```text
1. 固定模型、dtype、quantization、parallel config；
2. 单请求、短 prompt、max_tokens=1；
3. 关闭 logprobs / prompt_logprobs / grammar / spec decode / LoRA；
4. 固定 sampling 为 greedy；
5. 固定 attention backend；
6. 禁用或固定 cudagraph capture size；
7. 再逐个打开原问题中的功能；
8. 一旦复现，记录 batch token 数、num_reqs、dtype、backend、runtime mode。
```

对不同问题的最小复现重点：

```text
attention：构造相同 prompt length / decode step；
quantization：只跑包含目标 linear 的一次 forward；
MoE：构造会路由到目标 expert 的 token；
sampling：固定 logits 或固定 seed / top-k / top-p；
PP：固定 stage 数和 intermediate tensor shape；
DP：至少两个 rank，构造不同 token 数看 padding。
```

---

## 17. 性能异常的判断顺序

性能问题按下面顺序排查更稳：

```text
1. 看整体耗时落在 schedule / preprocess / forward / sample / bookkeep 哪段；
2. 如果 forward 慢，看 attention / linear / MoE / communication；
3. 如果 sample 慢，看 top-p、logprobs、grammar、spec decode；
4. 如果 bookkeep 慢，看 D2H copy 和 Python list 转换；
5. 如果 schedule 慢，看 KV allocation、prefix cache、structured output 等待；
6. 如果波动大，看 CUDA Graph runtime mode 是否频繁 NONE；
7. 如果多卡慢，看 collective kernel 和 rank 间 shape 是否一致。
```

不要只看平均吞吐。很多算子问题只在某类 step 出现：

```text
prefill 慢；
decode 慢；
first token 慢；
长上下文慢；
带 logprobs 慢；
MoE 某些 expert 热点慢；
DP padding 后某些 rank 空转。
```

---

## 18. 常见问题分类与入口

### 18.1 backend 不支持 dtype / shape

看：

```text
validate_configuration reasons
attention selector config
quant config supported dtype
kernel-specific shape constraints
```

### 18.2 optional dependency 缺失

看：

```text
ImportError
FlashInfer / aiter / Triton / CUTLASS / custom op import
fallback warning
```

### 18.3 CUDA extension 未编译或加载失败

看：

```text
vllm._C import
_custom_ops 调用栈
torch.ops.vllm.* 是否存在
平台是否是 CUDA / ROCm / XPU 对应 build
```

### 18.4 metadata 与 tensor shape 不一致

看：

```text
attention metadata
slot mapping
block table
scheduler_output.total_num_scheduled_tokens
num_tokens_padded
input_batch 状态
```

### 18.5 cache layout 错误

看：

```text
backend.get_required_kv_cache_layout()
set_kv_cache_layout(...)
KV cache group config
slot mapping by layer / by group
```

### 18.6 quantization scale 或 packed weight 不匹配

看：

```text
weight_loader shard offset
packed_dim / packed_factor
block quant scale shape
Marlin tile adjustment
checkpoint quant metadata
```

---

## 19. 实际排查 checklist

```text
1. 把报错或性能异常归到 attention / linear / MoE / sampling / communication / cudagraph；
2. 查日志确认实际 backend；
3. 记录 dtype、device capability、TP/PP/DP/EP/CP config；
4. 打印或断点检查关键 shape 和 padded shape；
5. 打开 profiler，看耗时最多的 range 和 kernel；
6. 判断是否 fallback 到 torch / native；
7. 关闭复杂功能构造最小复现；
8. 如果是多卡，确认所有 rank 的 shape、runtime mode、collective 顺序一致；
9. 如果是 CUDA Graph，确认 capture key 和 runtime batch descriptor；
10. 如果是 quantization，确认 weight shard、scale、zero point 与 packed layout。
```

---

## 20. 容易疑惑的点

### 20.1 没有报错但慢，是不是 kernel bug？

不一定。更常见是 fallback 到兼容慢路径，或者 logprobs / top-p / D2H copy / DP padding 增加了输出侧成本。

### 20.2 profiler 里看到 aten::sort 就一定错了吗？

不一定。top-p PyTorch fallback 会使用 sort，小 batch 或某些平台这是正常路径。但如果大 batch CUDA 上频繁出现，可能说明 optimized sampler 没选中。

### 20.3 CUDA Graph 开了，为什么 profiler 还有很多 kernel？

CUDA Graph replay 仍然执行 kernel，只是减少 launch overhead。要看 runtime mode、graph replay range 和 kernel launch 模式，而不是期待 kernel 消失。

### 20.4 多卡 hang 一定是 NCCL 问题吗？

不一定。很多 NCCL hang 是上游 shape / collective 顺序不一致导致的。例如某 rank 走了 cudagraph，另一 rank eager；某 rank 进入 all-reduce，另一 rank 因异常提前退出。

### 20.5 NaN 一定来自 attention 吗？

不一定。NaN 可能来自 quantized GEMM、RoPE scaling、FP8 KV cache、logits processor、penalty、sampling temperature，也可能是模型权重本身问题。

---

## 21. 总结

算子排查可以压缩成：

```text
现象归类
  → backend 确认
  → shape / dtype / layout / metadata 校验
  → profiler 对齐源码阶段
  → fallback / capture / communication 判断
  → 最小复现
```

如果只记住一句话：

```text
vLLM 算子问题的关键不是先看某个 kernel，而是先把“实际 backend、rank-local shape、metadata、parallel group、cudagraph mode”这五件事对齐；对齐之后，报错和性能异常通常会落到很小的范围内。
```
