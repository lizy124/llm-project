# 11. 算子如何适配 TP / PP / DP / EP / CP？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\communication_op.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\parallel_state.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\device_communicators\all2all.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\linear.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\vocab_parallel_embedding.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\logits_processor.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\fused_moe\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\dp_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\cp_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\`

这个问题关注：tensor parallel、pipeline parallel、data parallel、expert parallel、context parallel 如何改变算子的输入输出布局、通信边界和 backend 选择。

---

## 1. 一句话回答

并行策略会改变算子的 rank-local 输入规模、权重分片方式、KV cache / attention metadata 布局和通信边界，因此同一个模型层在不同 parallel config 下会变成“本地 kernel + collective communication”的组合，而不是单个全局算子。

最小链路是：

```text
ParallelConfig
  → distributed groups
  → layer / token / expert / sequence partition
  → rank-local operator execution
  → all-reduce / all-gather / reduce-scatter / all2all / send-recv
  → 恢复全局语义输出
```

可以先记成：

```text
算子负责 rank-local compute；
parallel group 负责把局部结果拼回全局语义。
```

---

## 2. vLLM 里有哪些并行维度

常见并行维度如下：

```text
TP：Tensor Parallelism
  按 hidden / head / vocab / weight 维度切分模型内部张量。

PP：Pipeline Parallelism
  按 layer 切分模型，stage 之间传 intermediate tensors。

DP：Data Parallelism
  多份模型副本处理不同 batch / request 分片，需要协调 padding、调度和 metrics。

EP：Expert Parallelism
  MoE expert 分布在不同 rank，请求 token 需要 dispatch 到目标 expert rank 再 combine。

CP：Context Parallelism
  按上下文 / sequence 维度切分 attention 相关计算和 KV cache 访问。
```

这些维度可以组合。例如：

```text
TP + PP：每个 pipeline stage 内部仍有 tensor parallel；
TP + EP：MoE expert GEMM 和 dense linear 都可能有通信；
DP + CUDA Graph：每个 DP rank 的 token 数需要协调到一致的 padded shape；
CP + attention backend：backend 必须支持返回额外 metadata 或支持 PCP / DCP。
```

---

## 3. 通信 primitive 的基础入口

TP 相关通信入口在：`vllm/distributed/communication_op.py`

```python
def tensor_model_parallel_all_reduce(input_):
    return get_tp_group().all_reduce(input_)

def tensor_model_parallel_all_gather(input_, dim=-1):
    return get_tp_group().all_gather(input_, dim)

def tensor_model_parallel_reduce_scatter(input_, dim=-1):
    return get_tp_group().reduce_scatter(input_, dim)

def tensor_model_parallel_gather(input_, dst=0, dim=-1):
    return get_tp_group().gather(input_, dst, dim)
```

位置：`vllm/distributed/communication_op.py:12`

这些函数背后是 `GroupCoordinator`，它封装了：

```text
device group / CPU group
NCCL / Gloo / platform communicator
all-reduce / all-gather / reduce-scatter
send / recv / tensor dict send-recv
MoE dispatch / combine
```

位置：`vllm/distributed/parallel_state.py`

---

## 4. TP 如何影响 linear 算子

TP 最典型地体现在 linear layer。

vLLM 里有三类常见 linear：

```text
ReplicatedLinear：每个 rank 有完整权重，无 TP 切分；
ColumnParallelLinear：按输出维度切分权重；
RowParallelLinear：按输入维度切分权重。
```

源码位置：`vllm/model_executor/layers/linear.py`

---

## 5. ColumnParallelLinear：切输出维度

`ColumnParallelLinear` 定义在：`linear.py:393`

数学形式：

```text
Y = X A + b
A = [A_1, A_2, ..., A_p]
rank i 计算 Y_i = X A_i
```

初始化时：

```python
self.output_size_per_partition = divide(output_size, self.tp_size)
self.output_partition_sizes = [self.output_size_per_partition]
```

位置：`vllm/model_executor/layers/linear.py:434`

forward 时：

```python
output_parallel = self.quant_method.apply(self, input_, bias)

if self.gather_output and self.tp_size > 1:
    output = tensor_model_parallel_all_gather(output_parallel)
else:
    output = output_parallel
```

位置：`vllm/model_executor/layers/linear.py:548`

因此 ColumnParallelLinear 的语义是：

```text
默认：每个 rank 只保留输出 hidden 的一个 shard；
gather_output=True：通过 all-gather 拼成完整输出。
```

典型使用场景：

```text
QKV projection
MLP gate_up projection
某些 fused projection
```

这些层的输出后面往往还能继续以 shard 形式消费，所以不一定立刻 all-gather。

---

## 6. RowParallelLinear：切输入维度

`RowParallelLinear` 定义在：`linear.py:1493`

数学形式：

```text
Y = X A + b
A 按输入维度切分：A = [A_1; A_2; ...; A_p]
X 按 hidden 维度切分：X = [X_1, X_2, ..., X_p]
rank i 计算 partial Y_i = X_i A_i
最终 Y = sum_i partial Y_i
```

初始化时：

```python
self.input_size_per_partition = divide(input_size, self.tp_size)
self.output_size_per_partition = output_size
```

位置：`vllm/model_executor/layers/linear.py:1543`

forward 时：

```python
if self.input_is_parallel:
    input_parallel = input_
else:
    split_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
    input_parallel = split_input[self.tp_rank].contiguous()

output_parallel = self.quant_method.apply(self, input_parallel, bias_)

if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
else:
    output = output_parallel
```

位置：`vllm/model_executor/layers/linear.py:1628`

这里有两个关键点：

```text
input_is_parallel=True：上游已经给了本 rank 的 hidden shard；
reduce_results=True：通过 all-reduce 把各 rank 的 partial output 求和。
```

bias 只在 rank 0 加一次：

```python
bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
```

位置：`vllm/model_executor/layers/linear.py:1641`

否则 TP>1 时 bias 会被重复累加。

---

## 7. MergedColumnParallelLinear 与 QKV / MLP fused weight

很多模型会把多个 projection 合并成一个权重，例如：

```text
QKV projection：q_proj + k_proj + v_proj
MLP gate_up projection：gate_proj + up_proj
```

`MergedColumnParallelLinear` 继承自 `ColumnParallelLinear`，但 `output_partition_sizes` 是多个 logical output size 的列表。

位置：`vllm/model_executor/layers/linear.py:577`

它的 weight loader 会按 shard id 分别处理：

```text
q / k / v shard
gate / up shard
quantized packed weight shard
Marlin / FP8 block scale shard
bitsandbytes 4bit shard
```

位置：`vllm/model_executor/layers/linear.py:662`

所以 TP 不只是 forward 时切 tensor，也影响权重加载、量化 scale、packed layout 的 offset 计算。

---

## 8. TP 与量化算子的组合

linear layer 的实际 GEMM 由：

```python
self.quant_method.apply(self, x, bias)
```

执行。

位置：`vllm/model_executor/layers/linear.py:217`

这意味着 TP 切分后的每个 rank-local matrix 仍然可以是：

```text
unquantized GEMM
FP8 GEMM
AWQ / GPTQ / Marlin
CUTLASS / scaled_mm
bitsandbytes
torch fallback
```

量化参数也要按 TP shard 对齐：

```text
qweight shard
scale shard
zero point shard
block quant scale shard
packed_factor offset
marlin tile offset
```

如果 shard size、group size、block size 和 TP size 不整除，就可能触发特殊处理、fallback 或报错。

---

## 9. TP 如何影响 vocab embedding

`VocabParallelEmbedding` 按 vocabulary 维度切分 embedding table。

定义位置：`vllm/model_executor/layers/vocab_parallel_embedding.py:197`

初始化时：

```python
self.num_embeddings_per_partition = divide(self.num_embeddings_padded, self.tp_size)
```

位置：`vllm/model_executor/layers/vocab_parallel_embedding.py:300`

forward 时：

```text
1. 每个 rank 根据自己的 vocab range 构造 input mask；
2. 不属于本 rank 的 token id 被映射并 mask；
3. 本 rank 做 embedding lookup；
4. 不属于本 rank 的位置输出置零；
5. TP all-reduce 求和，得到完整 embedding。
```

代码：

```python
masked_input, input_mask = get_masked_input_and_mask(...)
output_parallel = self.quant_method.embedding(self, masked_input.long())
output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
output = tensor_model_parallel_all_reduce(output_parallel)
```

位置：`vllm/model_executor/layers/vocab_parallel_embedding.py:472`

所以 embedding 的 TP 语义是：

```text
每个 rank 只拥有部分 vocab；
每个 token 只会在拥有该 vocab shard 的 rank 上产生非零 embedding；
all-reduce 后每个 rank 都得到完整 hidden state。
```

---

## 10. TP 如何影响 LM head 和 logits

`ParallelLMHead` 继承自 `VocabParallelEmbedding`。

定义位置：`vllm/model_executor/layers/vocab_parallel_embedding.py:504`

它不直接 forward：

```python
def forward(self, input_):
    raise RuntimeError("LMHead's weights should be used in the sampler.")
```

位置：`vllm/model_executor/layers/vocab_parallel_embedding.py:562`

原因是 LM head 权重通常被 logits processor / sampler 使用，输出 logits 是 vocab parallel 的：

```text
每个 TP rank 计算本 rank vocab shard 的 logits；
需要采样 / logprobs 时再 gather / reindex / mask padding；
最终恢复 token_id 与 vocab index 的对应关系。
```

`VocabParallelEmbedding.get_sharded_to_full_mapping()` 专门处理 TP gather 后的 vocab reindex：

```text
base embeddings
added embeddings
padding
```

位置：`vllm/model_executor/layers/vocab_parallel_embedding.py:365`

---

## 11. TP 如何影响 attention

attention 里的 TP 主要体现在：

```text
num_heads 切分；
num_kv_heads 切分或复制；
QKV projection 的 column parallel；
attention output projection 的 row parallel；
KV cache block table / slot mapping 按 rank-local head 布局；
backend selection 要看 head_size、num_heads、kv_cache_dtype、block_size。
```

常见链路是：

```text
hidden states
  → ColumnParallelLinear 得到 rank-local QKV
  → attention backend 只处理本 rank 的 heads
  → RowParallelLinear 输出投影
  → TP all-reduce 合并 partial hidden
```

这就是 Transformer block 中经典的：

```text
column parallel projection → local attention / MLP → row parallel projection → all-reduce
```

---

## 12. PP 如何改变 forward 边界

Pipeline Parallelism 按 layer 切分模型。不同 PP rank 不拥有完整模型层。

在 `GPUModelRunner` 里，PP 影响：

```text
first rank：接收 input ids / embeddings；
middle rank：接收上一 stage 的 IntermediateTensors；
last rank：产生 logits / sampler output；
非 last rank：通常只向下一 stage 发送 intermediate tensors。
```

`_preprocess()` 中可以看到：

```python
if is_first_rank:
    intermediate_tensors = None
else:
    assert intermediate_tensors is not None
    intermediate_tensors = self.sync_and_gather_intermediate_tensors(...)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3544`

PP stage 之间通信使用 `get_pp_group()` 的 tensor dict send / recv 机制。

`GroupCoordinator.send_tensor_dict()` 和 `recv_tensor_dict()` 支持：

```text
metadata 走 CPU group；
GPU tensor 走 device group；
可选 all_gather_group 优化；
irecv / isend handle；
record_stream 避免异步发送期间 tensor 被复用。
```

位置：`vllm/distributed/parallel_state.py:980`

---

## 13. PP 下 sample_tokens 的差异

generation 输出只在最后一个 PP rank 上真正完成。

非最后 rank 通常只负责：

```text
forward 当前 stage；
发送 intermediate tensors；
处理必要的 KV connector output；
在 async scheduling 下接收 / 转发 sampled token ids。
```

最后 rank 才会：

```text
hidden_states → logits → Sampler → ModelRunnerOutput
```

因此 PP 会把“模型算子链”切成 stage-local 子链，stage 边界就是 send / recv 算子。

---

## 14. DP 如何影响算子执行

Data Parallelism 下，每个 DP rank 有一份模型副本，通常处理不同 requests / tokens。

对单个算子来说：

```text
linear / attention / sampler 仍在本 rank 本地执行；
不同 DP rank 之间不需要为每个 layer all-reduce；
但调度、padding、CUDA Graph、metrics、负载均衡需要跨 DP rank 协调。
```

V1 中 DP batch 协调在：`vllm/v1/worker/dp_utils.py`

入口：

```python
def coordinate_batch_across_dp(...)
```

位置：`vllm/v1/worker/dp_utils.py:164`

它会通过 DP group all-reduce 同步：

```text
每个 rank 的原始 token 数；
每个 rank 的 padded token 数；
是否 microbatch；
cudagraph mode。
```

位置：`vllm/v1/worker/dp_utils.py:36`

---

## 15. DP 与 CUDA Graph padding

DP 下一个重要问题是：不同 rank 的 batch token 数可能不同。

如果某些 rank 使用 CUDA Graph，collective 和 graph replay 需要各 rank 的 shape / runtime mode 一致。

`_synchronize_dp_ranks()` 会：

```text
1. all-reduce 收集所有 DP rank 的 token / padding / cudagraph mode；
2. cudagraph mode 取 min，如果任一 rank 是 NONE，则所有 rank 同步为 NONE；
3. 如果启用 cudagraph 或 ubatching，则所有 rank pad 到最大 token 数；
4. 决定是否所有 rank 都使用 microbatch。
```

位置：`vllm/v1/worker/dp_utils.py:101`

所以 DP 对算子的影响不是权重切分，而是：

```text
本 rank 算子可能因为其他 DP rank 的 shape 而被 pad；
本 rank cudagraph mode 可能因为其他 rank 不能 replay 而降级。
```

---

## 16. EP 如何影响 MoE 算子

Expert Parallelism 主要作用在 MoE。

普通 dense MLP 是：

```text
hidden states → gate/up/down GEMM → output
```

MoE + EP 变成：

```text
hidden states
  → router logits / topk ids
  → dispatch token 到 expert owner rank
  → local expert GEMM / fused_moe kernel
  → combine expert outputs
  → 恢复 token 顺序
```

`GroupCoordinator` 对 EP 提供了几个方法：

```python
def dispatch_router_logits(hidden_states, router_logits, ...)
def dispatch(hidden_states, topk_weights, topk_ids, ...)
def combine(hidden_states, ...)
```

位置：`vllm/distributed/parallel_state.py:1186`

实际通信由 device communicator 实现，常见是 all2all。

源码位置：`vllm/distributed/device_communicators/all2all.py`

---

## 17. EP dispatch / combine 的算子边界

MoE 的 EP 边界可以理解为：

```text
router / grouped_topk
  → communication dispatch
  → local fused expert compute
  → communication combine
```

dispatch 需要携带：

```text
hidden_states
topk_weights
topk_ids
extra_tensors
sequence parallel 标记
```

位置：`vllm/distributed/parallel_state.py:1206`

这意味着 MoE kernel 的输入不再是“原 batch token 顺序”，而往往是按 expert / rank 重排后的 token buffer。

因此 fused MoE backend 要处理：

```text
token-expert mapping；
expert-local token count；
topk weights；
expert parallel rank；
all2all dispatch/combine layout；
quantized expert weight layout；
shared expert 分支。
```

---

## 18. EP 与 TP 的组合

MoE 层可以同时有：

```text
TP：每个 expert 的权重 / hidden 维度切分；
EP：不同 expert 分布到不同 rank；
DP：不同 batch shard；
SP：sequence parallel 下 token 维度也可能切分。
```

组合后，一个 MoE step 可能包含：

```text
router local compute
  → EP all2all dispatch
  → TP-local expert GEMM
  → TP all-reduce / reduce-scatter
  → EP all2all combine
```

这也是 MoE 性能排查时要同时看 compute kernel 和 communication kernel 的原因。

---

## 19. CP 如何影响 attention 算子

Context Parallelism 主要影响 attention / KV cache。

V1 里 CP 兼容性检查在：`vllm/v1/worker/cp_utils.py`

入口：

```python
def check_attention_cp_compatibility(vllm_config)
```

位置：`vllm/v1/worker/cp_utils.py:14`

它检查：

```text
prefill_context_parallel_size
decode_context_parallel_size
cp_kv_cache_interleave_size
attention layer impl capability
```

关键要求：

```text
DCP：attention implementation 必须在 decode 时返回 softmax LSE；
PCP：attention implementation 必须支持 PCP；
MTP + cp_kv_cache_interleave_size > 1：backend 需要显式支持。
```

代码：

```python
assert layer_impl.need_to_return_lse_for_decode
assert layer_impl.supports_pcp
assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size
```

位置：`vllm/v1/worker/cp_utils.py:24`

---

## 20. CP 为什么需要 LSE

Context parallel 会把 attention 的上下文切到不同 rank。

如果每个 rank 只看局部 context，就需要把局部 attention 结果合并成全局 softmax 结果。合并 softmax 时需要：

```text
局部 max / LSE
局部 attention output
跨 rank merge
```

因此 DCP decode 要求 attention backend 返回 LSE：

```text
没有 LSE，就无法正确 merge 不同 context shard 的 softmax 结果。
```

这也是 CP 和普通 TP attention 的一个关键区别。

---

## 21. 通信与计算如何交错

vLLM 并不是简单地每层都同步等待，而是尽量让通信和计算在可控范围内交错：

```text
PP send / recv 使用 isend / irecv；
tensor dict 传输后对 CUDA tensor record_stream；
MoE dispatch / combine 由 device communicator 封装；
DP padding 协调用小 tensor all-reduce；
async output copy 用独立 stream；
CUDA Graph 要求通信 shape 稳定。
```

但是有些同步无法避免：

```text
RowParallelLinear 的 all-reduce 之后才能得到完整 output；
VocabParallelEmbedding 的 all-reduce 之后才能得到完整 embedding；
PP 下一 stage 必须等上一 stage intermediate；
EP combine 之后才能恢复 MoE output；
CP merge 之后才能得到全局 attention 语义。
```

---

## 22. 并行策略对 backend selection 的影响

并行策略会改变 backend 选择输入：

```text
TP 改变 num_heads / local heads / hidden shard size；
TP 改变 linear weight shard shape 和 quantization shard size；
PP 改变某 rank 是否需要 logits / sampler；
DP 改变 cudagraph padding 和 microbatch 决策；
EP 改变 MoE backend、all2all backend、expert-local batch size；
CP 要求 attention backend 支持 PCP / DCP / LSE；
SP / ubatching 改变 token shape 和 collective 约束。
```

所以 backend fallback 有时不是硬件不支持，而是并行配置让 shape 或 metadata 进入了另一个约束区间。

---

## 23. CUDA Graph 下的并行限制

CUDA Graph 要求通信和计算的 shape 稳定。

并行场景下要额外保证：

```text
TP ranks 的 tensor shape 一致；
DP ranks 的 cudagraph mode 一致；
PP stage 的 send / recv tensor metadata 一致；
EP dispatch token buffer shape 可 capture 或 fallback；
CP attention metadata 和 LSE merge buffer shape 稳定；
LoRA active count / ubatch slices / padded num_tokens 命中 capture key。
```

`GPUModelRunner._determine_batch_execution_and_padding()` 会在 DP 场景下协调 cudagraph mode 和 padding：

```python
coordinate_batch_across_dp(...)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3882`

---

## 24. 一个 dense Transformer block 的 TP 时间线

```text
1. input hidden states 在每个 TP rank 上复制或已切分；
2. QKV ColumnParallelLinear：每 rank 计算部分 heads；
3. attention backend：每 rank 处理本地 heads 和 KV cache shard；
4. output RowParallelLinear：每 rank 计算 partial output；
5. TP all-reduce 得到完整 attention output；
6. MLP gate_up ColumnParallelLinear：每 rank 计算部分 intermediate；
7. activation fused op 本地执行；
8. down RowParallelLinear：每 rank 计算 partial hidden；
9. TP all-reduce 得到完整 block output。
```

---

## 25. 一个 MoE + EP 时间线

```text
1. hidden states 进入 router；
2. router logits 通过 topk 得到 topk_ids / topk_weights；
3. EP dispatch 根据 expert owner 重排并发送 token；
4. 每个 rank 对本地 expert token 运行 fused MoE / quantized expert GEMM；
5. EP combine 把 expert output 发回原 token owner；
6. 按 topk_weights 合并；
7. 如有 TP / SP，再做对应 collective；
8. 输出恢复到原 batch token 顺序。
```

---

## 26. 常见问题

### 26.1 为什么 ColumnParallelLinear 有时不 all-gather？

因为后续算子可能能直接消费 output shard。例如 QKV projection 后 attention 按 head 分片执行，不需要马上拼回完整 hidden。

### 26.2 为什么 RowParallelLinear 要 all-reduce？

因为每个 rank 只计算了输入 shard 对最终输出的 partial contribution，必须求和才能得到全局 output。

### 26.3 为什么 embedding 用 all-reduce 而不是 all-gather？

每个 token 只在拥有该 vocab shard 的 rank 上有非零 embedding，其他 rank 是零。all-reduce 求和后自然得到完整 embedding。

### 26.4 PP 非最后 rank 为什么没有 logits？

因为它只拥有前面或中间的 layer，输出是 intermediate tensors，logits / sampling 只在最后 stage 完成。

### 26.5 CP 为什么挑 attention backend？

因为 CP 需要 attention backend 支持局部 attention 结果的跨 rank merge，DCP 还要求 decode 返回 softmax LSE。

---

## 27. 总结

并行与算子的关系可以压缩成：

```text
TP：切权重 / hidden / vocab / heads，用 all-reduce 或 all-gather 恢复语义；
PP：切 layer，用 send / recv 传 intermediate tensors；
DP：复制模型，用 batch 协调和 padding 保持 runtime 一致；
EP：切 expert，用 all2all dispatch / combine 连接 router 和 expert GEMM；
CP：切 context，用 attention backend 的 LSE / merge 能力恢复全局 attention。
```

如果只记住一句话：

```text
vLLM 的并行算子不是单纯替换 kernel，而是把每个全局 layer 拆成 rank-local compute 和 collective communication 的组合，并让 backend selection、weight loading、KV cache layout、CUDA Graph padding 都服从这个组合。
```
