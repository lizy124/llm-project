# 03. Tensor Parallel 在 vLLM 中如何切分模型？

源码位置：

- `vllm/vllm/config/parallel.py`
- `vllm/vllm/config/vllm.py`
- `vllm/vllm/config/model.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/distributed/communication_op.py`
- `vllm/vllm/model_executor/layers/linear.py`
- `vllm/vllm/model_executor/layers/vocab_parallel_embedding.py`
- `vllm/vllm/model_executor/layers/logits_processor.py`
- `vllm/vllm/model_executor/parameter.py`
- `vllm/vllm/model_executor/models/llama.py`
- `vllm/vllm/model_executor/models/`

本问题关注：Tensor Parallel 在 vLLM 中切分哪些 tensor，QKV projection、MLP、embedding、lm_head 如何分片，每个 TP rank 计算哪一部分，forward 中哪些位置需要 all-reduce / all-gather，以及 TP 如何影响 attention head、KV head、GQA/MQA、logits 和权重加载。

---

## 1. 一句话回答

Tensor Parallel 是单个模型层内部的并行：

```text
同一层权重被切到多个 TP rank；
每个 rank 计算这一层的一部分结果；
必要时通过 TP group 内通信，把 partial result 合成下一层需要的形态。
```

在 vLLM 里最典型的模式是：

```text
ColumnParallelLinear：切输出维，不立即通信；
RowParallelLinear：切输入维，算完 all-reduce；
QKVParallelLinear：按 attention head 切 Q/K/V；
MergedColumnParallelLinear：把 gate/up 等多个 column-parallel 矩阵打包切；
VocabParallelEmbedding / ParallelLMHead：按 vocab 维切 embedding / lm_head；
LogitsProcessor：对 vocab shard 产生的 logits 做 gather / all-gather。
```

最小记忆：

```text
TP 是“一个 layer 内多个 rank 一起算”；Column 切输出，Row 切输入，Row 的末尾通常 all-reduce。
```

---

## 2. 本文要回答的问题

```text
TP group 是怎么创建的？
ColumnParallelLinear / RowParallelLinear 分别切什么？
QKV projection 如何随 TP 切分？
GQA/MQA 下 KV heads 少于 TP size 时怎么办？
MLP gate/up/down projection 如何通信？
embedding / vocab parallel / lm_head 如何处理？
TP 下 logits 和 sampling 前为什么需要 gather？
权重加载如何只加载本 rank 的 shard？
quantized linear 是否改变 TP 通信语义？
TP 和 PP / DP / EP / CP 的边界是什么？
```

---

## 3. TP 在 vLLM 总链路中的位置

可以把 TP 链路拆成 5 层：

```text
1. 配置层
   ParallelConfig.tensor_parallel_size

2. 校验层
   ModelConfig.verify_with_parallel_config()

3. group 层
   initialize_model_parallel() 创建 TP group

4. 模型层
   QKVParallelLinear / MergedColumnParallelLinear / RowParallelLinear / VocabParallelEmbedding

5. 通信层
   tensor_model_parallel_all_reduce / all_gather / gather / reduce_scatter
```

完整主线：

```text
VllmConfig
  → ParallelConfig.tensor_parallel_size
  → ModelConfig.verify_with_parallel_config()
  → initialize_model_parallel()
  → get_tp_group()
  → 模型构造 parallel linear / vocab parallel 层
  → 每个 TP rank 加载自己的权重 shard
  → forward 中各 rank 计算局部输出
  → RowParallelLinear / embedding / logits 等位置执行 TP 通信
```

---

## 4. ParallelConfig 中 TP 如何配置 world size

### 4.1 tensor_parallel_size 是并行维度之一

`ParallelConfig` 中：

```python
pipeline_parallel_size: int = Field(default=1, ge=1)
tensor_parallel_size: int = Field(default=1, ge=1)
prefill_context_parallel_size: int = Field(default=1, ge=1)
data_parallel_size: int = Field(default=1, ge=1)
```

位置：`vllm/vllm/config/parallel.py:117` 到 `vllm/vllm/config/parallel.py:127`

这里的 `tensor_parallel_size` 表示：

```text
每个模型层内部由多少个 rank 共同计算。
```

### 4.2 world_size 默认是 PP x TP x PCP

`ParallelConfig.__post_init__()` 中：

```python
self.world_size = (
    self.pipeline_parallel_size
    * self.tensor_parallel_size
    * self.prefill_context_parallel_size
)
```

位置：`vllm/vllm/config/parallel.py:791` 到 `vllm/vllm/config/parallel.py:797`

如果使用 `external_launcher`，还会把 DP 也乘进去：

```python
if self.distributed_executor_backend == "external_launcher":
    self.world_size *= self.data_parallel_size
```

位置：`vllm/vllm/config/parallel.py:799` 到 `vllm/vllm/config/parallel.py:801`

因此普通内部执行里可以先记成：

```text
worker world size = TP x PP x PCP
```

DP 在很多场景下是外层副本维度，不总是直接计入这个 worker world size。

### 4.3 DCP 复用 TP 维度

`decode_context_parallel_size` 的注释说明：

```text
world size does not change by dcp, it simply reuse the GPUs of TP group
```

位置：`vllm/vllm/config/parallel.py:339` 到 `vllm/vllm/config/parallel.py:342`

并且要求：

```python
if self.tensor_parallel_size % self.decode_context_parallel_size != 0:
    raise ValueError(...)
```

位置：`vllm/vllm/config/parallel.py:500` 到 `vllm/vllm/config/parallel.py:507`

这说明 DCP 是在 TP group 内继续切分的一种并行，不会增加总 worker 数。

---

## 5. 模型配置如何校验 TP 合法性

`VllmConfig.__post_init__()` 会调用：

```python
self.model_config.verify_with_parallel_config(self.parallel_config)
```

位置：`vllm/vllm/config/vllm.py:863` 到 `vllm/vllm/config/vllm.py:866`

### 5.1 attention heads 必须能被 TP 整除

`ModelConfig.verify_with_parallel_config()` 中：

```python
if total_num_attention_heads % tensor_parallel_size != 0:
    raise ValueError(...)
```

位置：`vllm/vllm/config/model.py:1159` 到 `vllm/vllm/config/model.py:1170`

原因是：

```text
Q heads 是按 TP rank 均匀切的；
如果不能整除，每个 rank 无法拿到相同数量的 query heads。
```

### 5.2 本地 Q heads / KV heads 如何计算

本地 Q heads：

```python
def get_num_attention_heads(self, parallel_config: ParallelConfig) -> int:
    num_heads = self.model_arch_config.total_num_attention_heads
    return num_heads // parallel_config.tensor_parallel_size
```

位置：`vllm/vllm/config/model.py:1272` 到 `vllm/vllm/config/model.py:1274`

本地 KV heads：

```python
def get_num_kv_heads(self, parallel_config: ParallelConfig) -> int:
    if self.use_mla:
        return 1

    total_num_kv_heads = self.get_total_num_kv_heads()
    return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
```

位置：`vllm/vllm/config/model.py:1259` 到 `vllm/vllm/config/model.py:1270`

关键区别：

```text
Q heads：必须整除 TP size；
KV heads：能切则切，少于 TP size 时每个 rank 至少保留 1 个 KV head。
```

### 5.3 DCP + GQA/MQA 有额外约束

当 `decode_context_parallel_size > 1` 且不是 MLA 时：

```python
assert tensor_parallel_size > total_num_kv_heads
max_dcp_size = tensor_parallel_size // total_num_kv_heads
assert decode_context_parallel_size <= max_dcp_size
num_q_per_kv = total_num_attention_heads // total_num_kv_heads
assert num_q_per_kv % decode_context_parallel_size == 0
```

位置：`vllm/vllm/config/model.py:1184` 到 `vllm/vllm/config/model.py:1207`

这说明：

```text
DCP 会复用 TP ranks；
在 GQA/MQA 下，还必须保持 Q heads 与 KV heads 的分组关系能继续被 DCP 整除。
```

---

## 6. TP group 是如何创建的

### 6.1 initialize_model_parallel 是核心入口

入口：

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
```

位置：`vllm/vllm/distributed/parallel_state.py:1694` 到 `vllm/vllm/distributed/parallel_state.py:1700`
核心区别可以这么记：

```text
TP / PP / PCP / DCP：模型内部执行拓扑，调用方显式传入 initialize_model_parallel()
DP：外层副本/请求并行维度，来自 VllmConfig.parallel_config.data_parallel_size
SP：这里没有作为独立 group 入参；sequence-parallel 路径通常复用已有 TP / DP / EP 通信域，在 PP residual 或 MoE dispatch/combine 中改变 token/activation 布局
```

注释给了一个例子：

```text
8 GPUs, TP=2, PP=4

TP groups:
  [g0, g1], [g2, g3], [g4, g5], [g6, g7]

PP groups:
  [g0, g2, g4, g6], [g1, g3, g5, g7]
```

位置：`vllm/vllm/distributed/parallel_state.py:1711` 到 `vllm/vllm/distributed/parallel_state.py:1718`

### 6.2 rank layout 是 ExternalDP x DP x PP x PCP x TP

代码中明确写了 layout：

```python
# the layout order is: ExternalDP x DP x PP x TP
```
注释滞后/简写了
实际 reshape 包含 PCP：

```python
all_ranks = torch.arange(world_size).reshape(
    -1,
    data_parallel_size,
    pipeline_model_parallel_size,
    prefill_context_model_parallel_size,
    tensor_model_parallel_size,
)
```

位置：`vllm/vllm/distributed/parallel_state.py:1760` 到 `vllm/vllm/distributed/parallel_state.py:1775`

可以理解为：

```text
TP 是 rank layout 的最后一维；
相邻的一组 TP ranks 共同计算同一个 pipeline stage 内的同一层。
```

### 6.3 TP group 的构造

构造 TP group 的代码：

```python
group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
group_ranks = [x.tolist() for x in group_ranks]
_TP = init_model_parallel_group(
    group_ranks,
    get_world_group().local_rank,
    backend,
    use_message_queue_broadcaster=True,
    group_name="tp",
)
```

位置：`vllm/vllm/distributed/parallel_state.py:1777` 到 `vllm/vllm/distributed/parallel_state.py:1792`

创建后，全局 `_TP` 保存这个 group。

### 6.4 获取 TP group / rank / world size

`parallel_state.py` 提供：

```python
def get_tp_group() -> GroupCoordinator:
    assert _TP is not None
    return _TP
```

位置：`vllm/vllm/distributed/parallel_state.py:1346` 到 `vllm/vllm/distributed/parallel_state.py:1351`

以及：

```python
def get_tensor_model_parallel_world_size() -> int:
    return get_tp_group().world_size

def get_tensor_model_parallel_rank() -> int:
    return get_tp_group().rank_in_group
```

位置：`vllm/vllm/distributed/parallel_state.py:2012` 到 `vllm/vllm/distributed/parallel_state.py:2019`

模型层不会自己算 global rank，而是通过这些 helper 知道：

```text
当前 TP group 有多大；
当前 rank 是 TP group 内第几个。
```

---

## 7. TP 通信原语封装在哪里

`communication_op.py` 是 TP 通信的薄封装：

```python
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_tp_group().all_reduce(input_)

def tensor_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return get_tp_group().all_gather(input_, dim)

def tensor_model_parallel_reduce_scatter(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return get_tp_group().reduce_scatter(input_, dim)

def tensor_model_parallel_gather(input_: torch.Tensor, dst: int = 0, dim: int = -1):
    return get_tp_group().gather(input_, dst, dim)
```

位置：`vllm/vllm/distributed/communication_op.py:12` 到 `vllm/vllm/distributed/communication_op.py:35`

它的意义是：

```text
模型层只表达“我要 TP all-reduce / all-gather”；
具体用 NCCL、自定义 communicator、CPU group 还是设备 group，由 GroupCoordinator 处理。
```

常见用途：

```text
all-reduce：RowParallelLinear、VocabParallelEmbedding 合并 partial result；
all-gather：ColumnParallelLinear 可选 gather_output、LogitsProcessor 某些平台 gather logits；
gather：默认 logits 只 gather 到 rank 0；
reduce-scatter：某些优化路径使用，普通 dense Llama 主线不一定走。
```

---

## 8. ColumnParallelLinear：切输出维

### 8.1 数学含义

`ColumnParallelLinear` 注释写得很清楚：

```text
Y = X A + b
A is parallelized along its second dimension as A = [A_1, ..., A_p].
```

位置：`vllm/vllm/model_executor/layers/linear.py:392` 到 `vllm/vllm/model_executor/layers/linear.py:416`

也就是说：

```text
输入 X 每个 rank 都有完整副本；
权重 A 按输出维切分；
每个 rank 只算一部分输出 Y_i = X A_i。
```

### 8.2 初始化时计算本 rank 输出大小

```python
self.tp_rank = get_tensor_model_parallel_rank()
self.tp_size = get_tensor_model_parallel_world_size()
self.input_size_per_partition = input_size
self.output_size_per_partition = divide(output_size, self.tp_size)
self.output_partition_sizes = [self.output_size_per_partition]
```

位置：`vllm/vllm/model_executor/layers/linear.py:420` 到 `vllm/vllm/model_executor/layers/linear.py:439`

因此权重形状大致是：

```text
[output_size / tp_size, input_size]
```

### 8.3 forward 默认不 all-gather

forward 中：

```python
output_parallel = self.quant_method.apply(self, input_, bias)

if self.gather_output and self.tp_size > 1:
    output = tensor_model_parallel_all_gather(output_parallel)
else:
    output = output_parallel
```

位置：`vllm/vllm/model_executor/layers/linear.py:548` 到 `vllm/vllm/model_executor/layers/linear.py:562`

默认 `gather_output=False`，所以：

```text
ColumnParallelLinear 通常让输出继续保持分片状态，交给后面的 RowParallelLinear 消费。
```

这减少了中间通信。

---

## 9. RowParallelLinear：切输入维，末尾 all-reduce

### 9.1 数学含义

`RowParallelLinear` 注释：

```text
Y = X A + b
A is parallelized along its first dimension and X along its second dimension
```

位置：`vllm/vllm/model_executor/layers/linear.py:1491` 到 `vllm/vllm/model_executor/layers/linear.py:1524`

也就是说：

```text
输入 X 已经按 hidden 维切成 X_i；
每个 rank 持有对应的 A_i；
每个 rank 算 partial output = X_i A_i；
最后 all-reduce 求和得到完整 Y。
```

### 9.2 初始化时切输入维

```python
self.input_size_per_partition = divide(input_size, self.tp_size)
self.output_size_per_partition = output_size
self.output_partition_sizes = [output_size]
```

位置：`vllm/vllm/model_executor/layers/linear.py:1528` 到 `vllm/vllm/model_executor/layers/linear.py:1548`

权重形状大致是：

```text
[output_size, input_size / tp_size]
```

### 9.3 input_is_parallel 控制是否先切输入

forward 中：

```python
if self.input_is_parallel:
    input_parallel = input_
else:
    split_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
    input_parallel = split_input[self.tp_rank].contiguous()
```

位置：`vllm/vllm/model_executor/layers/linear.py:1628` 到 `vllm/vllm/model_executor/layers/linear.py:1639`

常见主线里，前一个 column-parallel 层已经输出分片，所以 `input_is_parallel=True`。

### 9.4 reduce_results 控制是否 all-reduce

```python
output_parallel = self.quant_method.apply(self, input_parallel, bias_)

if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
else:
    output = output_parallel
```

位置：`vllm/vllm/model_executor/layers/linear.py:1640` 到 `vllm/vllm/model_executor/layers/linear.py:1650`

默认 `reduce_results=True`，所以：

```text
RowParallelLinear 通常是 TP dense block 中的通信汇合点。
```

### 9.5 bias 只在 rank 0 加一次

```python
bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
```

位置：`vllm/vllm/model_executor/layers/linear.py:1640` 到 `vllm/vllm/model_executor/layers/linear.py:1644`

原因是：

```text
如果每个 rank 都加 bias，再 all-reduce 会把 bias 加 tp_size 次。
```

---

## 10. MLP 中 TP 如何工作

以 Llama MLP 为例。

### 10.1 gate/up 使用 MergedColumnParallelLinear

```python
self.gate_up_proj = MergedColumnParallelLinear(
    input_size=hidden_size,
    output_sizes=[intermediate_size] * 2,
    ...
)
```

位置：`vllm/vllm/model_executor/models/llama.py:89` 到 `vllm/vllm/model_executor/models/llama.py:102`

`MergedColumnParallelLinear` 是打包版 column parallel：

```text
把多个输出矩阵沿输出维拼在一起；
每个 logical matrix 分别按 TP 切分；
常见用途是 gate_proj + up_proj。
```

它要求：

```python
assert all(output_size % self.tp_size == 0 for output_size in output_sizes)
```

位置：`vllm/vllm/model_executor/layers/linear.py:617` 到 `vllm/vllm/model_executor/layers/linear.py:622`

### 10.2 down 使用 RowParallelLinear

```python
self.down_proj = RowParallelLinear(
    input_size=intermediate_size,
    output_size=hidden_size,
    reduce_results=reduce_results,
    ...
)
```

位置：`vllm/vllm/model_executor/models/llama.py:103` 到 `vllm/vllm/model_executor/models/llama.py:110`

### 10.3 forward 主线

```python
x, _ = self.gate_up_proj(x)
x = self.act_fn(x)
x, _ = self.down_proj(x)
```

位置：`vllm/vllm/model_executor/models/llama.py:118` 到 `vllm/vllm/model_executor/models/llama.py:122`

通信模式：

```text
hidden states replicated on each TP rank
  → gate/up column-parallel：每 rank 算 intermediate shard，不通信
  → activation 在 shard 上本地执行
  → down row-parallel：每 rank 算 hidden partial
  → all-reduce 得到完整 hidden states
```

所以 MLP 的 TP 心智模型是：

```text
Column 切开，Row 合回。
```

---

## 11. Attention 中 TP 如何工作

以 Llama attention 为例。

### 11.1 本地 head 数计算

```python
tp_size = get_tensor_model_parallel_world_size()
self.total_num_heads = num_heads
assert self.total_num_heads % tp_size == 0
self.num_heads = self.total_num_heads // tp_size
self.total_num_kv_heads = num_kv_heads
```

位置：`vllm/vllm/model_executor/models/llama.py:125` 到 `vllm/vllm/model_executor/models/llama.py:148`

KV head 逻辑：

```python
if self.total_num_kv_heads >= tp_size:
    assert self.total_num_kv_heads % tp_size == 0
else:
    assert tp_size % self.total_num_kv_heads == 0
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

位置：`vllm/vllm/model_executor/models/llama.py:148` 到 `vllm/vllm/model_executor/models/llama.py:156`

含义：

```text
Q heads：按 TP 均分；
KV heads：够多就按 TP 均分，不够就复制。
```

### 11.2 QKVParallelLinear 按 head 维切 Q/K/V

Llama attention 中：

```python
self.qkv_proj = QKVParallelLinear(
    hidden_size=hidden_size,
    head_size=self.head_dim,
    total_num_heads=self.total_num_heads,
    total_num_kv_heads=self.total_num_kv_heads,
    ...
)
```

位置：`vllm/vllm/model_executor/models/llama.py:165` 到 `vllm/vllm/model_executor/models/llama.py:173`

`QKVParallelLinear` 注释说明：

```text
When the number of key/value heads is smaller than the number of query heads
(e.g., multi-query/grouped-query attention), the key/value head may be replicated
while the query heads are partitioned.
```

位置：`vllm/vllm/model_executor/layers/linear.py:914` 到 `vllm/vllm/model_executor/layers/linear.py:923`

### 11.3 QKVParallelLinear 的切分逻辑

```python
self.num_heads = divide(self.total_num_heads, tp_size)
if tp_size >= self.total_num_kv_heads:
    self.num_kv_heads = 1
    self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
else:
    self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
    self.num_kv_head_replicas = 1
```

位置：`vllm/vllm/model_executor/layers/linear.py:942` 到 `vllm/vllm/model_executor/layers/linear.py:973`

输出分组：

```python
self.output_sizes = [
    self.num_heads * self.head_size * tp_size,
    self.num_kv_heads * self.head_size * tp_size,
    self.num_kv_heads * self.v_head_size * tp_size,
]
```

位置：`vllm/vllm/model_executor/layers/linear.py:974` 到 `vllm/vllm/model_executor/layers/linear.py:984`

虽然 `output_sizes` 记录的是全局逻辑大小，但每个 rank 实际只拿其中一片。

### 11.4 forward 中切 q/k/v

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q, k = self.rotary_emb(positions, q, k)
attn_output = self.attn(q, k, v)
output, _ = self.o_proj(attn_output)
```

位置：`vllm/vllm/model_executor/models/llama.py:224` 到 `vllm/vllm/model_executor/models/llama.py:234`

其中：

```text
q_size = 本地 num_heads * head_dim
kv_size = 本地 num_kv_heads * head_dim
```

位置：`vllm/vllm/model_executor/models/llama.py:158` 到 `vllm/vllm/model_executor/models/llama.py:162`

### 11.5 o_proj 是 RowParallelLinear

```python
self.o_proj = RowParallelLinear(
    input_size=self.total_num_heads * self.head_dim,
    output_size=hidden_size,
    ...
)
```

位置：`vllm/vllm/model_executor/models/llama.py:175` 到 `vllm/vllm/model_executor/models/llama.py:181`

attention 输出在每个 rank 上对应本地 heads 的 shard，`o_proj` 把它映射回 hidden，并 all-reduce。

所以 attention 的 TP 主线是：

```text
hidden states replicated
  → qkv_proj column-parallel：每 rank 得到本地 Q heads / KV heads
  → attention 在本地 heads 上计算
  → o_proj row-parallel：partial hidden all-reduce
  → hidden states 再次 replicated
```

---

## 12. GQA / MQA 下 KV head 少于 TP size 时怎么办

这是 TP 中最容易混淆的点。

假设：

```text
num_attention_heads = 32
num_key_value_heads = 1
TP size = 4
```

那么：

```text
每个 rank 的 Q heads = 32 / 4 = 8
每个 rank 的 KV heads = max(1, 1 / 4) = 1
```

也就是：

```text
Q heads 被切分；
唯一的 KV head 被复制到每个 rank。
```

`QKVParallelLinear` 用：

```python
self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
```

位置：`vllm/vllm/model_executor/layers/linear.py:968` 到 `vllm/vllm/model_executor/layers/linear.py:973`

在权重加载时，K/V 的 shard rank 使用：

```python
if loaded_shard_id == "q":
    shard_rank = self.tp_rank
else:
    shard_rank = self.tp_rank // self.num_kv_head_replicas
```

位置：`vllm/vllm/model_executor/layers/linear.py:1278` 到 `vllm/vllm/model_executor/layers/linear.py:1286`

含义：

```text
Q：每个 TP rank 加载不同 shard；
K/V：多个 TP rank 可能加载同一个 KV shard，实现复制。
```

因此：

```text
TP size 可以大于 KV head 数，但不能随意大于 Q head 的可整除范围。
```

---

## 13. VocabParallelEmbedding：embedding 按 vocab 维切

### 13.1 vocab 会 padding 后再切

`VocabParallelEmbedding` 注释说明：

```text
Embedding parallelized in the vocabulary dimension.
we pad the vocabulary size to make sure it is divisible by the number of model parallel GPUs.
```

位置：`vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:196` 到 `vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:235`

padding 函数：

```python
def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to
```

位置：`vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:87` 到 `vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:89`

### 13.2 每个 rank 有自己的 vocab range

```python
self.num_embeddings_per_partition = divide(
    self.num_embeddings_padded, self.tp_size
)
```

位置：`vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:300` 到 `vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:315`

range 计算：

```python
padded_org_vocab_start_index, padded_org_vocab_end_index = (
    vocab_range_from_global_vocab_size(org_vocab_size_padded, tp_rank, tp_size)
)
```

位置：`vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:327` 到 `vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:363`

### 13.3 forward 用 mask + all-reduce 合并 embedding

```python
if self.tp_size > 1:
    masked_input, input_mask = get_masked_input_and_mask(...)
else:
    masked_input = input_

output_parallel = self.quant_method.embedding(self, masked_input.long())

if self.tp_size > 1:
    output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)

output = tensor_model_parallel_all_reduce(output_parallel)
```

位置：`vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:472` 到 `vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:492`

这表示：

```text
每个 rank 只对自己 vocab shard 内的 token 查 embedding；
不属于本 rank 的 token 输出置 0；
TP all-reduce 后，每个 token 只留下正确 rank 的 embedding。
```

所以 embedding 的 TP 模式是：

```text
vocab shard lookup → mask invalid token → all-reduce 得到完整 hidden states。
```

---

## 14. ParallelLMHead 和 logits 如何处理

### 14.1 ParallelLMHead 继承 VocabParallelEmbedding

```python
class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head."""
```

位置：`vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:503` 到 `vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:521`

LM head 权重和 embedding 类似，也是按 vocab 维切。

但它的 forward 不应该直接调用：

```python
def forward(self, input_):
    del input_
    raise RuntimeError("LMHead's weights should be used in the sampler.")
```

位置：`vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:562` 到 `vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:564`

### 14.2 LogitsProcessor 负责用 lm_head 计算 logits

`LogitsProcessor.forward()` 中：

```python
logits = self._get_logits(hidden_states, lm_head, embedding_bias)
```

位置：`vllm/vllm/model_executor/layers/logits_processor.py:54` 到 `vllm/vllm/model_executor/layers/logits_processor.py:73`

`_get_logits()`：

```python
logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
logits = self._gather_logits(logits)
if logits is not None:
    logits = logits[..., : self.org_vocab_size]
```

位置：`vllm/vllm/model_executor/layers/logits_processor.py:89` 到 `vllm/vllm/model_executor/layers/logits_processor.py:104`

### 14.3 logits gather / all-gather

```python
if self.use_all_gather:
    logits = tensor_model_parallel_all_gather(logits)
else:
    logits = tensor_model_parallel_gather(logits)
```

位置：`vllm/vllm/model_executor/layers/logits_processor.py:75` 到 `vllm/vllm/model_executor/layers/logits_processor.py:87`

含义：

```text
每个 TP rank 只算自己 vocab shard 的 logits；
采样前需要把各 rank 的 vocab logits 拼起来；
默认可能只 gather 到 rank 0，某些平台使用 all-gather 让每个 rank 都拿到完整 logits。
```

### 14.4 局部 argmax 优化

`get_top_tokens()` 提供一个优化：不 gather 全量 logits，只 gather 每个 rank 的 local top 值和 index。

```python
local_max_vals, local_max_indices = logits.max(dim=-1)
local_pair = torch.stack([local_max_vals.float(), global_indices.float()], dim=-1)
gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
```

位置：`vllm/vllm/model_executor/layers/logits_processor.py:106` 到 `vllm/vllm/model_executor/layers/logits_processor.py:156`

这个优化说明：

```text
vocab parallel 的 logits 通信可能很贵；
如果只需要 argmax，可以只通信每个 shard 的 top 候选。
```

---

## 15. 以 Llama 为例看完整 TP block

Llama 中的关键层：

```text
embed_tokens：VocabParallelEmbedding
qkv_proj：QKVParallelLinear
o_proj：RowParallelLinear
gate_up_proj：MergedColumnParallelLinear
down_proj：RowParallelLinear
lm_head：ParallelLMHead
logits_processor：LogitsProcessor
```

位置：`vllm/vllm/model_executor/models/llama.py:38` 到 `vllm/vllm/model_executor/models/llama.py:53`

一个 decoder layer 的 TP 主链路：

```text
hidden states replicated
  → self_attn.qkv_proj：column-parallel，得到本地 Q/K/V heads
  → attention：本地 heads 上计算 attention output shard
  → self_attn.o_proj：row-parallel，all-reduce 回 replicated hidden
  → mlp.gate_up_proj：merged column-parallel，得到本地 intermediate shard
  → activation：本地 shard 上执行
  → mlp.down_proj：row-parallel，all-reduce 回 replicated hidden
```

模型头部和尾部：

```text
input_ids
  → VocabParallelEmbedding：vocab shard lookup + all-reduce
  → transformer layers
  → ParallelLMHead：每 rank 算 vocab shard logits
  → LogitsProcessor：gather / all-gather logits
  → sampler
```

这就是 dense decoder-only 模型中最常见的 TP 执行图。

---

## 16. 权重加载如何适配 TP shard

TP 不只是 forward 切分，权重加载也必须只加载本 rank 对应的 shard。

### 16.1 ColumnParallelLinear 按 output_dim 切

```python
if output_dim is not None and not is_sharded_weight:
    shard_size = param_data.shape[output_dim]
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
```

位置：`vllm/vllm/model_executor/layers/linear.py:517` 到 `vllm/vllm/model_executor/layers/linear.py:538`

含义：

```text
Column parallel 的权重按输出维加载本 rank 分片。
```

### 16.2 RowParallelLinear 按 input_dim 切

```python
if input_dim is not None and not is_sharded_weight:
    shard_size = param_data.shape[input_dim]
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
```

位置：`vllm/vllm/model_executor/layers/linear.py:1597` 到 `vllm/vllm/model_executor/layers/linear.py:1617`

含义：

```text
Row parallel 的权重按输入维加载本 rank 分片。
```

### 16.3 QKVParallelLinear 按 q/k/v 分别切

QKV loader 支持 `loaded_shard_id` 为：

```text
q / k / v
```

位置：`vllm/vllm/model_executor/layers/linear.py:999` 到 `vllm/vllm/model_executor/layers/linear.py:1009`

每类 shard 的本地 offset / size：

```python
"q": self.num_heads * self.head_size
"k": self.num_kv_heads * self.head_size
"v": self.num_kv_heads * self.v_head_size
```

位置：`vllm/vllm/model_executor/layers/linear.py:1011` 到 `vllm/vllm/model_executor/layers/linear.py:1027`

Llama 的权重加载映射：

```python
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]
```

位置：`vllm/vllm/model_executor/models/llama.py:433` 到 `vllm/vllm/model_executor/models/llama.py:441`

这说明：

```text
checkpoint 里分开的 q_proj/k_proj/v_proj，会加载进 vLLM 内部 fused 的 qkv_proj；
checkpoint 里分开的 gate_proj/up_proj，会加载进 fused 的 gate_up_proj。
```

### 16.4 VocabParallelEmbedding 按 vocab range 加载

```python
start_idx = self.shard_indices.org_vocab_start_index
shard_size = self.shard_indices.org_vocab_end_index - start_idx
loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
param[loaded_weight.shape[0] :].data.fill_(0)
```

位置：`vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:430` 到 `vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:470`

含义：

```text
每个 TP rank 只加载自己 vocab shard 的 embedding / lm_head 权重；
padding 区域填 0。
```

---

## 17. quantized linear 是否改变 TP 语义

大多数情况下：

```text
quantization 改变的是权重存储格式、packing、scale 加载和 GEMM kernel；
不改变 Column / Row / QKV 的 TP 通信语义。
```

在 `linear.py` 中，所有 linear 都通过：

```python
self.quant_method.create_weights(...)
self.quant_method.apply(...)
```

来创建权重和执行计算。

位置：`vllm/vllm/model_executor/layers/linear.py:138` 到 `vllm/vllm/model_executor/layers/linear.py:176`

但 TP 切分仍由 layer 决定：

```text
ColumnParallelLinear：output_partition_sizes；
RowParallelLinear：input_size_per_partition；
QKVParallelLinear：q/k/v shard offset 和 size；
VocabParallelEmbedding：vocab shard range。
```

量化路径会额外处理：

```text
packed_dim
packed_factor
BlockQuantScaleParameter
Marlin shard offset
bitsandbytes 4bit shard
FP8 block scale shape mismatch
```

例如 ColumnParallelLinear 里有：

```python
if isinstance(param, BlockQuantScaleParameter):
    shard_size, shard_offset = adjust_block_scale_shard(...)
```

位置：`vllm/vllm/model_executor/layers/linear.py:70` 到 `vllm/vllm/model_executor/layers/linear.py:91`

所以结论是：

```text
量化让“怎么切权重 shard”更复杂，但不改变“Column 切输出、Row 切输入、Row all-reduce”的主语义。
```

---

## 18. TP 与 PP / DP / EP / CP 的关系

### 18.1 TP + PP

```text
PP 把层分到不同 pipeline stage；
TP 在每个 stage 内继续把单层 tensor 切到多个 rank。
```

rank layout 中 TP 是最后一维，PP 是更外层维度：

```text
... x PP x PCP x TP
```

因此常见组合是：

```text
每个 PP stage 有自己的 TP group；
同一个 TP group 内的 ranks 计算同一层的不同 tensor shard。
```

### 18.2 TP + DP

```text
DP 是模型副本维度；
每个 DP replica 内部都有自己的 TP/PP group。
```

也就是说：

```text
不同 DP replica 处理不同请求或 batch；
同一个 DP replica 内部的 TP ranks 合作计算同一个 batch。
```

### 18.3 TP + EP

`ParallelConfig` 中：

```python
enable_expert_parallel: bool = False
"""Use expert parallelism instead of tensor parallelism for MoE layers."""
```

位置：`vllm/vllm/config/parallel.py:160` 到 `vllm/vllm/config/parallel.py:164`

含义是：

```text
dense 层仍然可以用 TP；
MoE expert 层可以改用 EP 分专家，而不是仅用 TP 切 dense matrix。
```

### 18.4 TP + DCP / PCP

DCP / PCP 属于 context parallel，和 attention / KV cache 关系更强。

对 TP 来说关键是：

```text
DCP 复用 TP group 内的 ranks；
TP size 必须能被 DCP size 整除；
GQA/MQA 还要满足 Q-per-KV 的整除约束。
```

---

## 19. 几个容易疑惑的点

### 19.1 ColumnParallelLinear 会不会自动 all-reduce？

不会。

```text
ColumnParallelLinear 默认只产生本 rank 的 output shard；
只有 gather_output=True 时才 all-gather。
```

普通 transformer block 里通常让后续 RowParallelLinear 消费这个 shard。

### 19.2 RowParallelLinear 为什么要 all-reduce？

因为每个 rank 只算了：

```text
X_i A_i
```

完整结果是：

```text
sum_i X_i A_i
```

所以需要 TP all-reduce 求和。

### 19.3 attention 的 QKV projection 后为什么不 gather？

因为 attention 可以在本地 heads 上独立计算。

```text
每个 rank 拿一部分 Q heads；
attention 输出也是这部分 heads 的 shard；
o_proj 是 RowParallelLinear，会在输出投影后 all-reduce。
```

### 19.4 embedding 为什么是 all-reduce，而不是 all-gather？

embedding 是按 vocab 切的。

对每个 token：

```text
只有拥有该 token vocab range 的 rank 输出非零 embedding；
其他 rank 输出 0；
all-reduce 后得到正确 embedding。
```

因此 all-reduce 足够。

### 19.5 lm_head 为什么需要 gather logits？

lm_head 也是按 vocab 切的。

```text
每个 rank 只得到一段 vocab logits；
采样通常需要完整 vocab 分布；
所以 LogitsProcessor 需要 gather / all-gather。
```

### 19.6 TP size 能不能任意设置？

不能。

常见约束包括：

```text
num_attention_heads 必须能被 TP size 整除；
某些 MLP intermediate_size / fused output_sizes 要能被 TP size 整除；
vocab padded 后按 TP size 切；
TP size 必须能被 DCP size 整除；
GQA/MQA + DCP 有额外整除关系。
```

### 19.7 TP 和 KV cache 有什么关系？

TP 会影响每个 rank 本地 attention head / KV head 数，因此影响每个 rank 的 KV cache head 维。

```text
MHA：KV heads 通常随 TP 切分；
GQA：KV heads 可能随 TP 切分；
MQA 或 KV heads < TP size：KV heads 在多个 TP rank 间复制。
```

KV cache 详细关系可和 `08_kv_cache_and_parallelism.md`、`09_attention_and_parallelism.md` 联动阅读。

---

## 20. 最终可以记成一张表

| 模块 | 切分维度 | 每 rank 计算 | 通信 |
|---|---|---|---|
| `ColumnParallelLinear` | output dim | `Y_i = X A_i` | 默认无；可选 all-gather |
| `MergedColumnParallelLinear` | 多个 output dim | gate/up 等各自 shard | 默认无；可选 all-gather |
| `QKVParallelLinear` | Q/K/V head dim | 本地 Q heads / KV heads | 默认无 |
| `RowParallelLinear` | input dim | `X_i A_i` partial output | 默认 all-reduce |
| `VocabParallelEmbedding` | vocab dim | 本 shard embedding，其他 token 置 0 | all-reduce |
| `ParallelLMHead` | vocab dim | 本 shard logits | 由 LogitsProcessor gather |
| `LogitsProcessor` | vocab logits | 拼接 vocab shards | gather / all-gather |

通信语义：

| 通信 | 用途 |
|---|---|
| all-reduce | 合并 partial hidden states 或 masked embedding |
| all-gather | 拼接分片输出，让所有 rank 拿到完整 tensor |
| gather | 拼接分片输出，通常只给 dst rank |
| reduce-scatter | 聚合后继续保持分片的优化路径 |

---

## 21. 总结

Tensor Parallel 在 vLLM 中的主线可以压缩成：

```text
ParallelConfig 设置 TP size
  → initialize_model_parallel 创建 TP group
  → 模型层通过 get_tensor_model_parallel_rank/world_size 得到本 rank 信息
  → ColumnParallelLinear / QKVParallelLinear / MergedColumnParallelLinear 切输出维
  → RowParallelLinear 切输入维并 all-reduce
  → VocabParallelEmbedding / ParallelLMHead 切 vocab 维
  → LogitsProcessor gather vocab logits
```

如果只记住一句话：

```text
vLLM 的 dense TP 基本是 Megatron 风格：Column 切开中间维度，Row 在输出处 all-reduce 合回；attention heads、MLP intermediate、vocab logits 都围绕这个模式做特化。
```

最小执行图：

```text
Embedding：
  vocab shard lookup → mask → all-reduce

Attention：
  qkv column shard → local-head attention → o_proj row all-reduce

MLP：
  gate/up merged column shard → activation → down row all-reduce

Logits：
  lm_head vocab shard → gather / all-gather → sampling
```

TP 的核心边界是：

```text
它不决定调度哪些请求，不决定 pipeline stage，也不决定数据副本；
它只决定同一个模型层内部的 tensor 如何被多个 rank 分片计算，以及在哪里通信合并。
```
