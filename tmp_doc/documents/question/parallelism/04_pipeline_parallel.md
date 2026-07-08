# 04. Pipeline Parallel 如何切分 layer 和传递中间状态？

源码位置：

- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/distributed/utils.py`
- `vllm/vllm/sequence.py`
- `vllm/vllm/model_executor/models/utils.py`
- `vllm/vllm/model_executor/models/llama.py`
- `vllm/vllm/model_executor/model_loader/`
- `vllm/vllm/v1/executor/multiproc_executor.py`
- `vllm/vllm/v1/worker/gpu_worker.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/forward_context.py`

本文关注：Pipeline Parallel 如何把模型 layers 切成多个 stage，每个 PP rank 持有哪些层，stage 之间如何传递 hidden states / residual / intermediate tensors，哪些 rank 负责 logits / sampling，以及 PP 如何和 TP / DP / SP / KV cache 组合。

---

## 1. 一句话回答

Pipeline Parallel 是“按层切分模型”的并行方式：

```text
一个完整模型的 transformer layers 被切成多个 pipeline stage；
每个 PP rank 只加载并执行自己负责的连续 layer 区间；
非首 stage 不直接吃 input_ids，而是接收上一个 stage 的 IntermediateTensors；
非末 stage 不产出 logits，而是把 IntermediateTensors 发送给下一个 stage；
末 stage 才执行 norm / lm_head / logits / sampling 所需的最终输出。
```

最小主链路是：

```text
SchedulerOutput
  → Executor 向所有 worker 广播 execute_model
  → 每个 PP stage 都更新本地 batch / KV cache / attention metadata
  → PP first rank 用 input_ids / inputs_embeds 开始 forward
  → 非 last rank 返回 IntermediateTensors
  → Worker 通过 PP group isend_tensor_dict 发给下个 stage
  → 非 first rank 通过 irecv_tensor_dict 接收 IntermediateTensors
  → PP last rank 得到最终 hidden_states
  → compute_logits / sample_tokens
  → ModelRunnerOutput 从最后一个 PP stage 的首个 TP rank 返回
```

一句话记忆：

```text
TP 是“每层内部横向切张量”，PP 是“把层纵向切成多段串起来跑”。
```

---

## 2. 本文要回答的问题

```text
1. PP group 是如何初始化的？
2. PP rank / TP rank / DP rank 的 rank mesh 如何理解？
3. 模型 layers 如何分配到不同 PP stage？
4. first stage / middle stage / last stage 分别加载什么模块？
5. PPMissingLayer 是什么，为什么权重加载要跳过 missing 参数？
6. forward 时 input_ids、hidden_states、residual 如何在 stage 间传递？
7. IntermediateTensors 是什么？
8. Worker / ModelRunner 分别负责哪部分 PP 通信？
9. logits / sampling / ModelRunnerOutput 在哪个 rank 发生？
10. PP 和 TP / DP / SP / KV cache / multimodal 有什么关系？
```

---

## 3. PP 的整体心智模型

可以先把 PP 理解成三类 stage：

```text
PP first rank：
  - 持有输入 embedding；
  - 接收 input_ids / inputs_embeds；
  - 执行自己负责的前几层；
  - 输出 IntermediateTensors 给下一个 stage。

PP middle rank：
  - 不持有输入 embedding；
  - 接收上一个 stage 的 IntermediateTensors；
  - 执行自己负责的中间层；
  - 输出新的 IntermediateTensors 给下一个 stage。

PP last rank：
  - 接收上一个 stage 的 IntermediateTensors；
  - 执行最后几层；
  - 持有最终 norm / lm_head / logits_processor；
  - 产生最终 hidden_states / logits / sampling 输入。
```

对应到典型 Llama 模型：

```text
rank 0 / first stage:
  embed_tokens + layers[start:end]

rank 1..N-2 / middle stage:
  layers[start:end]

rank N-1 / last stage:
  layers[start:end] + norm + lm_head + logits_processor
```

源码中 `LlamaModel` 和 `LlamaForCausalLM` 就是这个模式：

- first rank 创建 `embed_tokens`：`vllm/vllm/model_executor/models/llama.py:365`
- 每个 rank 用 `make_layers()` 创建自己的 layer 区间：`vllm/vllm/model_executor/models/llama.py:375`
- last rank 创建 `norm`：`vllm/vllm/model_executor/models/llama.py:380`
- last rank 创建 `lm_head / logits_processor`：`vllm/vllm/model_executor/models/llama.py:518`

---

## 4. PP group 是如何初始化的

PP group 在 `initialize_model_parallel()` 中创建。

位置：`vllm/vllm/distributed/parallel_state.py:1694`

### 4.1 rank mesh 的维度顺序

vLLM 的模型并行 rank layout 注释写得很关键：

```text
ExternalDP x DP x PP x PCP x TP
```

对应代码：`vllm/vllm/distributed/parallel_state.py:1760` 到 `vllm/vllm/distributed/parallel_state.py:1775`

也就是说，world ranks 会先 reshape 成一个多维网格：

```text
[external_dp, dp, pp, pcp, tp]
```

然后不同并行维度通过 transpose / reshape 得到自己的通信组。

### 4.2 PP group 的构造方式

PP group 的构造在：

```text
all_ranks.transpose(2, 4)
  .reshape(-1, pipeline_model_parallel_size)
```

对应代码：`vllm/vllm/distributed/parallel_state.py:1835` 到 `vllm/vllm/distributed/parallel_state.py:1851`

它表达的是：

```text
固定 external_dp / dp / pcp / tp，
沿 PP 维度取出一组 rank，
这些 rank 组成一条 pipeline。
```

### 4.3 一个简单例子

源码注释给了 TP=2、PP=4、world_size=8 的例子：

```text
TP groups:
  [g0, g1], [g2, g3], [g4, g5], [g6, g7]

PP groups:
  [g0, g2, g4, g6], [g1, g3, g5, g7]
```

对应代码：`vllm/vllm/distributed/parallel_state.py:1711` 到 `vllm/vllm/distributed/parallel_state.py:1718`

含义是：

```text
同一个 PP group 内的 rank 拥有相同 TP rank 位置，
但处在不同 pipeline stage。
```

换句话说：

```text
TP group 横向切同一层；
PP group 纵向串不同层。
```

---

## 5. GroupCoordinator 提供哪些 PP 能力

PP group 本质是一个 `GroupCoordinator`。

位置：`vllm/vllm/distributed/parallel_state.py:351`

### 5.1 first / last / next / prev

`GroupCoordinator` 提供这些属性：

```text
first_rank
last_rank
is_first_rank
is_last_rank
next_rank
prev_rank
```

对应代码：`vllm/vllm/distributed/parallel_state.py:544` 到 `vllm/vllm/distributed/parallel_state.py:576`

这些属性是模型和 Worker 判断 PP 角色的基础：

```text
get_pp_group().is_first_rank
get_pp_group().is_last_rank
```

### 5.2 tensor dict 发送和接收

PP stage 之间传递的不是一个裸 tensor，而是 tensor dict。

发送接口：

```text
send_tensor_dict(...)
isend_tensor_dict(...)
```

接收接口：

```text
recv_tensor_dict(...)
irecv_tensor_dict(...)
```

对应代码：

- send：`vllm/vllm/distributed/parallel_state.py:941`
- isend：`vllm/vllm/distributed/parallel_state.py:979`
- recv：`vllm/vllm/distributed/parallel_state.py:1036`
- irecv：`vllm/vllm/distributed/parallel_state.py:1074`

### 5.3 为什么传 tensor dict

`IntermediateTensors` 可能包含多项张量，例如：

```text
hidden_states
residual
```

有些模型还可能包含更多中间状态。

`send_tensor_dict()` 会先把 dict 拆成：

```text
metadata_list：key、dtype、shape、device 等元数据
tensor_list：真正要发送的 tensor
```

对应代码：`vllm/vllm/distributed/parallel_state.py:82` 到 `vllm/vllm/distributed/parallel_state.py:104`

这样接收端可以先知道每个 tensor 的 shape / dtype，再分配接收 buffer。

### 5.4 PP 通信和 TP all-gather 优化

`send_tensor_dict()` / `irecv_tensor_dict()` 支持传入：

```text
all_gather_group=get_tp_group()
all_gather_tensors={...}
```

对应代码：`vllm/vllm/distributed/parallel_state.py:941` 到 `vllm/vllm/distributed/parallel_state.py:964`

作用是：

```text
如果某个 tensor 在 TP rank 间可以分片发送，
接收端再通过 TP all_gather 还原，
就可以减少跨 PP stage 的点对点发送量。
```

sequence parallelism 下，`residual` 是否可这样处理要额外判断，后面会展开。

---

## 6. layer 是如何切到 PP stage 的

模型通常通过 `make_layers()` 构建 transformer layers。

位置：`vllm/vllm/model_executor/models/utils.py:640`

### 6.1 get_pp_indices 计算 start/end

`make_layers()` 会调用：

```text
get_pp_indices(num_hidden_layers, pp_rank, pp_size)
```

对应代码：`vllm/vllm/model_executor/models/utils.py:656` 到 `vllm/vllm/model_executor/models/utils.py:662`

`get_pp_indices()` 定义在：

```text
vllm/vllm/distributed/utils.py:109
```

它返回：

```text
(start_layer, end_layer)
```

表示当前 PP rank 负责的 layer 区间：

```text
[start_layer, end_layer)
```

### 6.2 默认尽量均匀切分

如果没有手动指定，默认逻辑是：

```text
layers_per_partition = num_hidden_layers // pp_size
partitions = [layers_per_partition] * pp_size
```

如果不能整除，剩余 layers 会被分配到除 last partition 外的若干 stage，尽量平衡首尾 embedding / norm / lm_head 的额外开销。

对应代码：`vllm/vllm/distributed/utils.py:138` 到 `vllm/vllm/distributed/utils.py:149`

### 6.3 VLLM_PP_LAYER_PARTITION 手动指定

如果设置环境变量：

```text
VLLM_PP_LAYER_PARTITION=10,12,10
```

则表示 3 个 PP stage 分别持有 10、12、10 层。

源码会校验：

```text
len(partitions) == pp_size
sum(partitions) == num_hidden_layers
```

对应代码：`vllm/vllm/distributed/utils.py:125` 到 `vllm/vllm/distributed/utils.py:136`

### 6.4 PPMissingLayer 占位

`make_layers()` 返回的 `ModuleList` 长度仍然等于 `num_hidden_layers`，但不属于当前 stage 的 layer 会被替换成 `PPMissingLayer`：

```text
[PPMissingLayer() for _ in range(start_layer)]
+ 当前 rank 真实 layers
+ [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
```

对应代码：`vllm/vllm/model_executor/models/utils.py:664` 到 `vllm/vllm/model_executor/models/utils.py:670`

这样做的好处是：

```text
1. layer index 和原模型保持一致；
2. 参数名仍然能和 checkpoint 对齐；
3. forward 时只遍历当前 stage 的真实 [start_layer, end_layer)；
4. 权重加载时可以跳过 missing layer 参数。
```

---

## 7. PPMissingLayer 和权重加载

`PPMissingLayer` 定义在：

```text
vllm/vllm/model_executor/models/utils.py:627
```

它是一个占位层，forward 时直接返回输入。

### 7.1 为什么需要 PPMissingLayer

PP 下每个 rank 只持有部分层，但 checkpoint 里仍然有完整模型的参数。

如果当前 rank 不负责某一层，就不应该为它分配真实权重，也不应该加载对应参数。

因此 vLLM 用 `PPMissingLayer` 表示：

```text
这个模块在模型结构位置上存在，
但当前 PP rank 不实际持有它的参数和计算。
```

### 7.2 权重加载如何跳过 missing layer

工具函数：

```text
get_pp_missing_layer_names(model)
is_pp_missing_parameter(name, model)
```

定义位置：`vllm/vllm/model_executor/models/utils.py:679` 到 `vllm/vllm/model_executor/models/utils.py:705`

典型模型加载权重时会检查：

```text
if is_pp_missing_parameter(name, self):
    continue
```

Llama 例子见：

- stacked 参数路径：`vllm/vllm/model_executor/models/llama.py:464`
- 普通参数路径：`vllm/vllm/model_executor/models/llama.py:476`

所以 PP 下每个 stage 只加载本 stage 真实存在的层和必要的 embedding / norm / lm_head。

---

## 8. IntermediateTensors 是什么

`IntermediateTensors` 定义在：

```text
vllm/vllm/sequence.py:12
```

它本质上是：

```text
dict[str, torch.Tensor]
```

并提供了 `__getitem__`、slice、`items()` 等访问方式。

典型内容是：

```text
{
  "hidden_states": hidden_states,
  "residual": residual,
}
```

Llama 中对应：

```text
return IntermediateTensors(
  {"hidden_states": hidden_states, "residual": residual}
)
```

位置：`vllm/vllm/model_executor/models/llama.py:422` 到 `vllm/vllm/model_executor/models/llama.py:425`

### 8.1 为什么需要 residual

很多 decoder layer 使用 residual 延迟合并 / fused norm 的结构。

stage 间只传 `hidden_states` 不够，还要把 residual 一起传给下一个 stage，否则下个 stage 无法继续保持和单卡模型一致的计算图。

所以 PP 中间状态通常是：

```text
hidden_states + residual + 模型特定的额外中间张量
```

### 8.2 make_empty_intermediate_tensors_factory

模型会注册一个工厂函数，用于创建空的 intermediate buffer：

```text
make_empty_intermediate_tensors_factory(["hidden_states", "residual"], hidden_size)
```

对应代码：`vllm/vllm/model_executor/models/utils.py:708` 到 `vllm/vllm/model_executor/models/utils.py:721`

Llama 中注册位置：`vllm/vllm/model_executor/models/llama.py:385`

这个工厂在 dummy run / CUDA graph / 非首 PP rank 准备 buffer 时会用到。

---

## 9. 模型 forward 如何区分 PP stage

以 `LlamaModel.forward()` 为例。

位置：`vllm/vllm/model_executor/models/llama.py:392`

### 9.1 first rank：从 input_ids / inputs_embeds 开始

如果是 PP first rank：

```text
if get_pp_group().is_first_rank:
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.embed_input_ids(input_ids)
    residual = None
```

对应代码：`vllm/vllm/model_executor/models/llama.py:400` 到 `vllm/vllm/model_executor/models/llama.py:405`

这说明只有 first stage 真正消费 token ids 或外部 embeddings。

### 9.2 非 first rank：从 intermediate_tensors 恢复状态

如果不是 first rank：

```text
assert intermediate_tensors is not None
hidden_states = intermediate_tensors["hidden_states"]
residual = intermediate_tensors["residual"]
```

对应代码：`vllm/vllm/model_executor/models/llama.py:406` 到 `vllm/vllm/model_executor/models/llama.py:409`

所以非首 stage 的 forward 输入不是 token，而是上一个 stage 的中间状态。

### 9.3 每个 rank 只执行自己的 layer 区间

forward 中只遍历：

```text
islice(self.layers, self.start_layer, self.end_layer)
```

对应代码：`vllm/vllm/model_executor/models/llama.py:412` 到 `vllm/vllm/model_executor/models/llama.py:417`

这就是 `make_layers()` 返回 `start_layer / end_layer` 的运行时作用。

### 9.4 非 last rank：返回 IntermediateTensors

如果不是 PP last rank：

```text
return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
```

对应代码：`vllm/vllm/model_executor/models/llama.py:422` 到 `vllm/vllm/model_executor/models/llama.py:425`

### 9.5 last rank：做最终 norm 并返回 hidden_states

如果是 last rank：

```text
hidden_states, _ = self.norm(hidden_states, residual)
return hidden_states
```

对应代码：`vllm/vllm/model_executor/models/llama.py:427` 到 `vllm/vllm/model_executor/models/llama.py:431`

这也是为什么 logits / sampling 只能在 last stage 正常发生。

---

## 10. Worker.execute_model() 如何传递 PP 中间状态

PP 的跨 stage 通信主要在 `Worker.execute_model()` 中完成。

位置：`vllm/vllm/v1/worker/gpu_worker.py:836`

### 10.1 先等待上一轮异步发送完成

每轮开始前，Worker 会先等待上一次 PP send 完成：

```text
if self._pp_send_work:
    handle.wait()
    self._pp_send_work = []
```

对应代码：`vllm/vllm/v1/worker/gpu_worker.py:839` 到 `vllm/vllm/v1/worker/gpu_worker.py:843`

这样可以避免上一轮中间状态还没发完，本轮就复用或覆盖 buffer。

### 10.2 非 first rank 先发起 irecv

如果本轮有 forward，且当前不是 PP first rank：

```text
get_pp_group().irecv_tensor_dict(
  all_gather_group=get_tp_group(),
  all_gather_tensors=all_gather_tensors,
)
```

对应代码：`vllm/vllm/v1/worker/gpu_worker.py:881` 到 `vllm/vllm/v1/worker/gpu_worker.py:887`

返回结果会被包装成：

```text
AsyncIntermediateTensors
```

对应代码：`vllm/vllm/v1/worker/gpu_worker.py:889` 到 `vllm/vllm/v1/worker/gpu_worker.py:893`

### 10.3 AsyncIntermediateTensors 延迟等待通信完成

`AsyncIntermediateTensors` 定义在：

```text
vllm/vllm/v1/worker/gpu_worker.py:85
```

它持有：

```text
tensor_dict
comm_handles
comm_postprocess
```

只有当访问 `.tensors` 时，才会等待通信完成并执行 postprocess：

```text
wait_for_comm()
```

对应代码：`vllm/vllm/v1/worker/gpu_worker.py:99` 到 `vllm/vllm/v1/worker/gpu_worker.py:114`

这样可以让通信和本地输入准备尽量重叠。

### 10.4 调用 ModelRunner

Worker 把 `intermediate_tensors` 传给 ModelRunner：

```text
output = self.model_runner.execute_model(
  scheduler_output,
  intermediate_tensors,
)
```

对应代码：`vllm/vllm/v1/worker/gpu_worker.py:895` 到 `vllm/vllm/v1/worker/gpu_worker.py:898`

### 10.5 如果返回 IntermediateTensors，就发送给下个 stage

如果 ModelRunner 返回的是 `IntermediateTensors`，说明当前 rank 不是最后一个 PP stage。

Worker 会执行：

```text
self._pp_send_work = get_pp_group().isend_tensor_dict(
  output.tensors,
  all_gather_group=get_tp_group(),
  all_gather_tensors=all_gather_tensors,
)
return None
```

对应代码：`vllm/vllm/v1/worker/gpu_worker.py:910` 到 `vllm/vllm/v1/worker/gpu_worker.py:924`

所以：

```text
模型层负责返回 IntermediateTensors；
Worker 负责把 IntermediateTensors 发给下一段 pipeline。
```

---

## 11. ModelRunner._preprocess() 如何处理 PP 输入

`GPUModelRunner._preprocess()` 是模型 forward 前最后一层输入整形。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3430`

### 11.1 first rank 准备 token 输入

first rank 会走正常输入路径：

```text
input_ids / inputs_embeds / positions / model_kwargs
```

多模态、prompt embeds、encoder-decoder 等特殊输入也主要在 first rank 或对应 encoder 路径处理。

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:3451` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:3536`

### 11.2 非 first rank 同步并拷贝 intermediate_tensors

`_preprocess()` 中有关键分支：

```text
if is_first_rank:
    intermediate_tensors = None
else:
    assert intermediate_tensors is not None
    intermediate_tensors = self.sync_and_gather_intermediate_tensors(
        num_input_tokens, intermediate_tensors, True
    )
```

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:3547` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:3553`

这说明非首 PP rank 会把收到的中间张量同步到 ModelRunner 自己的持久 buffer，再切出本轮需要的 view。

### 11.3 sync_and_gather_intermediate_tensors 做什么

函数位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3285`

它主要做两件事：

```text
1. 如果 sync_self=True，把收到的 intermediate_tensors 拷贝到 self.intermediate_tensors 持久 buffer。
2. 返回只覆盖 num_tokens 的 IntermediateTensors view。
```

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:3296` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:3313`

如果启用了 sequence parallelism，且 `residual` 被 TP rank 分片，它还会先通过 TP group all_gather 还原 residual。

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:3293` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:3306`

---

## 12. ModelRunner.execute_model() 中 PP 如何影响输出

`GPUModelRunner.execute_model()` 负责本 rank 的模型 forward 和后处理。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4047`

### 12.1 所有 PP stage 都会准备本地执行状态

即使不是 first / last rank，当前 stage 仍然需要：

```text
_update_states()
_prepare_inputs()
_get_slot_mappings()
_build_attention_metadata()
_preprocess()
set_forward_context()
_model_forward()
```

原因是每个 stage 都可能包含 attention layers，都需要自己的：

```text
KV cache
slot mapping
attention metadata
positions
forward context
```

这点很重要：

```text
PP 只切模型层，不切请求调度协议。
SchedulerOutput 会广播到所有 PP stage。
```

### 12.2 forward 调用仍统一传 intermediate_tensors

真正模型调用在：

```text
self._model_forward(
  input_ids=input_ids,
  positions=positions,
  intermediate_tensors=intermediate_tensors,
  inputs_embeds=inputs_embeds,
  **model_kwargs,
)
```

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:4323` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4329`

模型内部再根据 `get_pp_group().is_first_rank / is_last_rank` 决定如何使用这些输入。

### 12.3 非 last rank 返回 IntermediateTensors

forward 后，如果不是 last rank：

```text
assert isinstance(hidden_states, IntermediateTensors)
self.kv_connector_output = kv_connector_output
return hidden_states
```

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:4340` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4346`

这会回到 Worker，由 Worker 发给下一个 PP stage。

### 12.4 last rank 才 compute_logits

如果是 last rank，生成模型会：

```text
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:4357` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4358`

然后保存 `ExecuteModelState`，等待 `sample_tokens()` 继续采样。

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:4389` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4408`

---

## 13. logits / sampling / ModelRunnerOutput 在哪个 rank

### 13.1 普通 PP 模式

普通 PP 模式下：

```text
非 last rank：
  execute_model() 返回 IntermediateTensors 给 Worker；
  Worker 发送给下个 stage；
  对 Executor 返回 None。

last rank：
  execute_model() 计算 logits；
  返回 None 等待 sample_tokens()；
  sample_tokens() 生成 ModelRunnerOutput。
```

也就是说用户侧最终输出只来自 last PP stage。

### 13.2 MultiprocExecutor 只收最后 PP stage 的一个 TP rank

`MultiprocExecutor._get_output_rank()` 明确写了：

```text
Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
```

对应代码：`vllm/vllm/v1/executor/multiproc_executor.py:495` 到 `vllm/vllm/v1/executor/multiproc_executor.py:509`

计算方式是：

```text
world_size - tensor_parallel_size * prefill_context_parallel_size
```

含义是：

```text
最后一个 PP stage 中的第一个 TP rank 负责向 Executor 返回 ModelRunnerOutput。
```

### 13.3 external_launcher 的 broadcast_pp_output

`GPUModelRunner` 初始化时有特殊模式：

```text
broadcast_pp_output = distributed_executor_backend == "external_launcher"
                      and pp_world_size > 1
```

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:472` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:479`

这种模式下，last rank 会把 logits broadcast 给其他 PP rank，保证 torchrun external launcher 下各 PP rank 同步。

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:4359` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4387`

但文档主线可以先记普通模式：

```text
last PP stage 负责 logits / sampling 输出。
```

---

## 14. PP 和 KV cache 的关系

PP 切的是 layer，所以 KV cache 也天然按 layer 切到各 stage。

### 14.1 每个 stage 只需要自己 attention layers 的 KV cache

因为当前 PP rank 只持有自己的真实 layers，所以它只需要这些 layers 对应的 KV cache。

非本 stage 的 layer 是 `PPMissingLayer`，不会有真实 attention 计算，也不需要对应 KV cache。

### 14.2 每个 stage 仍然要消费同一轮 SchedulerOutput

虽然 KV cache 的层不同，但请求维度是一致的。

每个 stage 都要知道：

```text
哪些请求在 batch 中；
每个请求本轮多少 token；
positions 是什么；
slot_mapping 是什么；
block_table 是什么；
哪些请求结束或抢占。
```

所以所有 PP stage 都会执行 `_update_states()` 和 attention metadata 准备。

### 14.3 KV connector 输出需要穿过 PP 边界

非 last rank 如果 forward 后拿到 `kv_connector_output`，会先存在：

```text
self.kv_connector_output
```

然后返回 IntermediateTensors。

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:4342` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4346`

如果后续 `sample_tokens()` 没有 execute state，也会构造 KV connector only output。

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:4429` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4437`

---

## 15. PP 和 TP / SP 的关系

### 15.1 PP + TP

PP 和 TP 常组合使用：

```text
PP：把 layers 切到不同 stage。
TP：每个 stage 内部，把单层参数和计算切到多个 TP rank。
```

通信上：

```text
PP stage 间：send/recv IntermediateTensors。
TP rank 间：all_reduce / all_gather / reduce_scatter 等张量并行通信。
```

PP 的 tensor dict 收发接口支持 `all_gather_group=get_tp_group()`，就是为了配合 TP 组优化 stage 间发送。

### 15.2 PP + SP

如果开启 sequence parallelism，某些中间状态可能在 TP rank 上按 token 维度分片。

Worker 会提前计算：

```text
all_gather_tensors = {
  "residual": not is_residual_scattered_for_sp(...)
}
```

对应代码：`vllm/vllm/v1/worker/gpu_worker.py:852` 到 `vllm/vllm/v1/worker/gpu_worker.py:879`

ModelRunner 接收后，如果发现 residual 是 scattered 的，会执行 TP all_gather 还原：

```text
v = get_tp_group().all_gather(v[:local_len], dim=0)
```

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:3293` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:3306`

所以：

```text
SP 改变 IntermediateTensors 中某些 tensor 的分布方式；
PP 通信需要知道哪些 tensor 能切片发送、哪些必须完整发送。
```

---

## 16. PP 和 DP 的关系

DP 是复制多套模型并行组来处理不同请求；PP 是一套模型内部的 layer 切分。

在 rank mesh 中：

```text
ExternalDP x DP x PP x PCP x TP
```

说明每个 DP replica 内部都可以有自己的 PP pipeline 和 TP groups。

可以理解成：

```text
DP replica 0:
  PP stage 0..N，每个 stage 内有 TP ranks

DP replica 1:
  PP stage 0..N，每个 stage 内有 TP ranks

...
```

Scheduler / Executor 层会保证同一个 DP group 中的 worker 协调执行，避免并行维度之间死锁。

---

## 17. PP 和多模态 / prompt embeds / encoder-decoder 的关系

`GPUModelRunner._preprocess()` 中的多模态和 prompt embeds 处理主要发生在 first PP rank：

```text
if supports_mm_inputs and is_first_rank and not is_encoder_decoder:
  执行 multimodal encoder / gather mm embeddings / 构造 inputs_embeds

elif enable_prompt_embeds and is_first_rank:
  处理 prompt embeds
```

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:3451` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:3528`

原因是：

```text
只有 first PP stage 直接接触 token ids / inputs_embeds；
后续 PP stage 只接收 hidden_states / residual。
```

encoder-decoder 模型会在 `_preprocess()` 中额外执行 encoder 并把 `encoder_outputs` 放进 `model_kwargs`。

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:3555` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:3562`

---

## 18. PP 下的一轮执行时序

把前面的源码串起来，一轮 PP forward 可以写成：

```text
Executor.collective_rpc("execute_model", SchedulerOutput)
  → 所有 Worker 同时收到 SchedulerOutput

每个 Worker:
  → 等待上一轮 PP send 完成
  → 如果不是 first PP rank，先 irecv_tensor_dict()
  → 调 GPUModelRunner.execute_model(...)

GPUModelRunner:
  → _update_states()
  → _prepare_inputs()
  → _get_slot_mappings()
  → _build_attention_metadata()
  → _preprocess()
      first rank: input_ids / inputs_embeds
      non-first rank: sync intermediate_tensors
  → set_forward_context(...)
  → _model_forward(...)

模型 forward:
  first rank: embed input_ids
  non-first rank: 读取 IntermediateTensors
  all ranks: 执行自己的 [start_layer, end_layer)
  non-last rank: 返回 IntermediateTensors
  last rank: 返回 final hidden_states

Worker / ModelRunner 后处理:
  non-last rank:
    → Worker isend_tensor_dict(intermediate_tensors)
    → 返回 None

  last rank:
    → compute_logits
    → execute_model 返回 None
    → sample_tokens
    → ModelRunnerOutput
```

---

## 19. 容易混淆的点

### 19.1 PP 不是 microbatch pipeline 调度

这里的 PP 主要是按 layer 切 stage，并在每一轮 token batch 上串行传递 intermediate tensors。

源码里也有 TODO 提到 external launcher 下尚未支持 overlapping micro-batches：

```text
TODO: Support overlapping micro-batches
```

对应代码：`vllm/vllm/v1/worker/gpu_model_runner.py:472` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:475`

### 19.2 非 first rank 也会准备 input metadata

非 first rank 不使用 input_ids 做 embedding，但仍然需要：

```text
positions
attention metadata
slot mapping
KV cache state
```

因为它自己的 layers 里也可能有 attention。

### 19.3 非 last rank 返回 None 给上层不代表没执行

非 last rank 的 ModelRunner 会返回 `IntermediateTensors` 给 Worker，Worker 发送后对 Executor 返回 `None`。

所以 `None` 只是表示：

```text
当前 rank 不产生最终 ModelRunnerOutput。
```

不是没有 forward。

### 19.4 IntermediateTensors 不是最终 hidden_states

`IntermediateTensors` 是 stage 间协议，通常包含：

```text
hidden_states
residual
```

只有 last rank 执行 final norm 后返回的 hidden_states，才会进入 logits / sampling。

### 19.5 PPMissingLayer 不是跳过执行区间的主要机制

真正执行哪些层由：

```text
islice(self.layers, self.start_layer, self.end_layer)
```

决定。

`PPMissingLayer` 更多是为了保持模块结构和参数名对齐，并让权重加载能识别哪些参数不属于当前 PP rank。

---

## 20. 最终可以记成一张表

| 阶段 | 关键代码 | 核心产物 | 作用 |
|---|---|---|---|
| 初始化并行组 | `initialize_model_parallel()` | TP / DCP / PCP / PP / DP / EP / EPLB groups | 建立 rank mesh 和 PP 通信组 |
| 计算 layer 区间 | `get_pp_indices()` | `start_layer / end_layer` | 决定当前 PP rank 负责哪些层 |
| 创建模型层 | `make_layers()` | `ModuleList` + `PPMissingLayer` | 只创建本 stage 真实 layers，其他位置占位 |
| 加载权重 | `is_pp_missing_parameter()` | 跳过 missing 参数 | 避免当前 stage 加载不属于自己的权重 |
| stage 间容器 | `IntermediateTensors` | hidden_states / residual | 表示 PP stage 间传递的中间状态 |
| 非首 rank 接收 | `irecv_tensor_dict()` | `AsyncIntermediateTensors` | 异步接收上一 stage 输出 |
| 输入预处理 | `_preprocess()` | input_ids 或 intermediate_tensors | first rank 用 token，非 first rank 用中间状态 |
| 模型 forward | model `forward()` | hidden_states 或 IntermediateTensors | 执行当前 stage 的 layer 区间 |
| 非末 rank 发送 | `isend_tensor_dict()` | PP send handles | 把中间状态发给下个 stage |
| 末 rank logits | `compute_logits()` | logits | 只在 last PP stage 产生采样输入 |
| 采样输出 | `sample_tokens()` | `ModelRunnerOutput` | 由最后 PP stage 的输出 rank 返回 Scheduler |

---

## 21. 一句话总结

Pipeline Parallel 在 vLLM 中是一套跨模型构造、权重加载、Worker 通信和 ModelRunner 执行的协议：

```text
初始化时：
  rank mesh 生成 PP group；
  make_layers 根据 pp_rank 切出当前 stage 的 layer 区间；
  不属于当前 stage 的层用 PPMissingLayer 占位并跳过权重加载。

执行时：
  first stage 从 input_ids / inputs_embeds 开始；
  中间 stage 接收 IntermediateTensors；
  每个 stage 执行自己的 layers 和 KV cache；
  非 last stage 发送 IntermediateTensors；
  last stage 计算 logits 并完成 sampling 输出。
```

如果只记住一句话，就是：

```text
PP 把“一个完整 forward”拆成多段模型层执行，段与段之间用 IntermediateTensors 串起来，最终只有 last PP stage 产出 logits / ModelRunnerOutput。
```
