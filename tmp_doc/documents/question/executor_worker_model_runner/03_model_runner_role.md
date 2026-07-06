# 03. ModelRunner 在 vLLM V1 里负责什么？

源码位置：

- `code/vllm/vllm\v1\worker\gpu_model_runner.py`
- `code/vllm/vllm\v1\worker\gpu_input_batch.py`
- `code/vllm/vllm\v1\worker\gpu_worker.py`
- `code/vllm/vllm\v1\core\sched\output.py`
- `code/vllm/vllm\v1\outputs.py`

本问题关注：`ModelRunner` 在 vLLM V1 执行层中的定位，它如何消费 `SchedulerOutput`，如何维护 worker 侧请求状态和 `InputBatch`，如何准备模型输入、attention metadata、KV cache slot mapping，如何执行 forward / logits / pooling / sampling，以及它和 `Worker`、`Scheduler`、`Executor` 的职责边界。

---

## 0. 梳理规划

参考 `scheduler` 目录的文档风格，本篇按“先定角色，再走主链路，再拆关键阶段，最后总结边界”的方式梳理 `ModelRunner`。

要回答的问题分成 10 组：

```text
1. ModelRunner 是哪一层？
2. 它和 Worker / Executor / Scheduler 的关系是什么？
3. 它初始化时保存哪些配置、状态和 persistent buffers？
4. 它如何加载模型、LoRA、drafter、MoE、CUDA graph wrapper？
5. 它如何初始化 KV cache、attention backend、metadata builders、InputBatch？
6. execute_model() 的主流程是什么？
7. _update_states() 如何把 SchedulerOutput 合并进 CachedRequestState / InputBatch？
8. _prepare_inputs() / _build_attention_metadata() / _preprocess() 如何准备模型输入？
9. _model_forward()、compute_logits、pooling、sample_tokens() 分别在哪里发生？
10. ModelRunnerOutput 如何产生并返回 Scheduler？
```

阅读顺序建议：

```text
02_worker_role.md
  → 03_model_runner_role.md
  → 04_execute_model_flow.md
  → 05_input_batch_and_state_update.md
  → 06_prepare_inputs_and_attention_metadata.md
  → 07_model_forward_and_logits.md
  → 08_sampling_and_model_runner_output.md
  → 09_worker_kv_cache_interaction.md
```

本篇重点讲定位和总链路，不会把每个 helper 函数展开到最细。后续专题再分别细拆：

```text
InputBatch / 状态更新 → 05_input_batch_and_state_update.md
输入准备 / attention metadata → 06_prepare_inputs_and_attention_metadata.md
forward / logits → 07_model_forward_and_logits.md
sampling / ModelRunnerOutput → 08_sampling_and_model_runner_output.md
KV cache 交互 → 09_worker_kv_cache_interaction.md
```

---

## 1. 一句话回答

`ModelRunner` 是 Worker 内部真正把 `SchedulerOutput` 变成模型执行的组件。

它不负责：

```text
不负责接用户请求；
不负责调度 waiting / running 队列；
不负责决定 token budget；
不负责跨 worker RPC；
不负责最终 detokenize 或构造 RequestOutput。
```

它负责：

```text
加载模型；
维护 worker 侧请求状态；
维护 InputBatch；
初始化和使用 KV cache；
根据 SchedulerOutput 准备 input_ids / positions / inputs_embeds；
构造 attention metadata / slot mapping / block table；
执行模型 forward；
计算 logits 或 pooling output；
应用 grammar bitmask；
调用 sampler；
处理 speculative decoding drafter / rejection；
生成 ModelRunnerOutput。
```

最小主线是：

```text
SchedulerOutput
  → GPUModelRunner._update_states()
  → InputBatch / CachedRequestState
  → GPUModelRunner._prepare_inputs()
  → GPUModelRunner._build_attention_metadata()
  → GPUModelRunner._preprocess()
  → GPUModelRunner._model_forward()
  → compute_logits / pooling
  → GPUModelRunner.sample_tokens()
  → ModelRunnerOutput
```

一句话压缩：

```text
Worker 管设备生命周期，ModelRunner 管一次 batch 如何真正跑进模型。
```

---

## 2. ModelRunner 在执行链路中的位置

从上游看，EngineCore 调用的是 Executor：

```text
EngineCore.step()
  → model_executor.execute_model(scheduler_output)
  → Executor.collective_rpc("execute_model")
  → Worker.execute_model(scheduler_output)
  → ModelRunner.execute_model(scheduler_output, intermediate_tensors)
```

`Worker.execute_model()` 中真正调用 ModelRunner 的代码是：

```python
output = self.model_runner.execute_model(
    scheduler_output, intermediate_tensors
)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:867` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:870`

因此组件关系是：

```text
Executor：负责 RPC / 多 worker 分发
Worker：负责本 rank / 本 device 的生命周期和控制面
ModelRunner：负责本 rank / 本 device 上的 batch 执行细节
```

ModelRunner 的核心输入是：

```text
SchedulerOutput
intermediate_tensors  # PP 非 first rank 时来自前一个 pipeline stage
```

核心输出可能是：

```text
ModelRunnerOutput
AsyncModelRunnerOutput
IntermediateTensors
None
```

不同返回值含义后面会展开。

---

## 3. GPUModelRunner 类定义和 mixin

GPU ModelRunner 定义为：

```python
class GPUModelRunner(
    LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin
):
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:418` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:420`

这说明 GPUModelRunner 本身集成了几类能力：

```text
LoRAModelRunnerMixin：LoRA 动态适配能力；
KVConnectorModelRunnerMixin：KV transfer / 外部 KV cache 交互；
ECConnectorModelRunnerMixin：Encoder cache transfer 交互；
GPUModelRunner 本体：模型加载、输入准备、forward、sampling、KV cache、attention metadata。
```

所以 `ModelRunner` 不是一个只负责 `forward()` 的薄封装，而是执行层最重的对象之一。

---

## 4. 初始化时保存哪些配置和状态

`GPUModelRunner.__init__()` 保存了几乎所有执行相关配置：

```python
self.vllm_config = vllm_config
self.model_config = vllm_config.model_config
self.cache_config = vllm_config.cache_config
self.offload_config = vllm_config.offload_config
self.compilation_config = vllm_config.compilation_config
self.lora_config = vllm_config.lora_config
self.load_config = vllm_config.load_config
self.parallel_config = vllm_config.parallel_config
self.scheduler_config = vllm_config.scheduler_config
self.speculative_config = vllm_config.speculative_config
self.observability_config = vllm_config.observability_config
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:426` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:436`

### 4.1 模型和 batch 上限

初始化时会记录：

```python
self.max_model_len = model_config.max_model_len
self.max_num_tokens = scheduler_config.max_num_batched_tokens
self.max_num_reqs = scheduler_config.max_num_seqs
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:461` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:468`

这三个上限会影响：

```text
InputBatch 大小；
persistent buffer 大小；
CUDA graph capture size；
最大 batch token 数；
最大并发 request 数。
```

### 4.2 模型相关信息

ModelRunner 还保存：

```python
self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
self.inputs_embeds_size = model_config.get_inputs_embeds_size()
self.use_alibi = model_config.uses_alibi
self.cascade_attn_enabled = not self.model_config.disable_cascade_attn
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:479` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:485`

这些信息会在 attention metadata 构造、输入准备和 forward 中使用。

### 4.3 多模态支持

多模态相关字段：

```python
self.mm_registry = MULTIMODAL_REGISTRY
self.uses_mrope = model_config.uses_mrope
self.uses_xdrope_dim = model_config.uses_xdrope_dim
self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(model_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:488` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:494`

如果是 encoder-decoder 模型，还会设置 encoder 长度：

```python
if self.model_config.is_encoder_decoder:
    self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
else:
    self.max_encoder_len = 0
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:496` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:501`

### 4.4 Sampler

采样器在初始化时创建：

```python
self.sampler = Sampler(
    logprobs_mode=self.model_config.logprobs_mode,
    use_fp64_gumbel=self.model_config.use_fp64_gumbel,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:506` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:510`

这说明：

```text
Sampler 属于 ModelRunner 侧，
不是 Scheduler，也不是 OutputProcessor。
```

Scheduler 只负责告诉 Worker 本轮该跑哪些请求；真正从 logits 到 token 的采样在 ModelRunner 里。

---

## 5. ModelRunner 内部维护哪些核心状态

### 5.1 请求状态

ModelRunner 保存 worker 侧请求状态：

```python
self.requests: dict[str, CachedRequestState] = {}
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:637` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:638`

这里的 `CachedRequestState` 定义在 `gpu_input_batch.py`，核心字段包括：

```python
@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    generator: torch.Generator | None

    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:32` 到 `code/vllm/vllm/v1/worker/gpu_input_batch.py:44`

它表示：

```text
某个请求在 Worker / ModelRunner 侧的持久化状态。
```

### 5.2 InputBatch

ModelRunner 初始化时会创建 `InputBatch`：

```python
self.input_batch = InputBatch(
    max_num_reqs=self.max_num_reqs,
    max_model_len=max(self.max_model_len, self.max_encoder_len),
    max_num_batched_tokens=self.max_num_tokens,
    device=self.device,
    pin_memory=self.pin_memory,
    vocab_size=self.model_config.get_vocab_size(),
    ...
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:661` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:689`

`InputBatch` 保存当前 persistent batch 的张量和索引状态：

```text
req_id_to_index
token_ids_cpu
is_token_ids
num_tokens_no_spec
num_prompt_tokens
num_computed_tokens
block_table
sampling metadata
pooling metadata
```

其中 `InputBatch` 的关键字段包括：

```python
self._req_ids: list[str | None] = []
self.req_id_to_index: dict[str, int] = {}
self.token_ids_cpu_tensor = torch.zeros((max_num_reqs, max_model_len), ...)
self.num_computed_tokens_cpu_tensor = torch.zeros((max_num_reqs,), ...)
self.block_table = MultiGroupBlockTable(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:126` 到 `code/vllm/vllm/v1/worker/gpu_input_batch.py:181`

可以这样理解：

```text
CachedRequestState：请求级状态
InputBatch：batch 级张量视图
```

### 5.3 KV cache 和 attention 相关状态

初始化时 KV cache 还没真正创建，只先准备字段：

```python
self.kv_caches: list[torch.Tensor] = []
self.cross_layers_kv_cache: torch.Tensor | None = None
self.cross_layers_attn_backend: type[AttentionBackend] | None = None
self.attn_groups: list[list[AttentionGroup]] = []
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:522` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:530`

真正初始化发生在 `initialize_kv_cache()`。

### 5.4 persistent buffers

ModelRunner 还预分配了一批 GPU buffer，用于 CUDA graph 和减少重复分配：

```python
self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
self.positions = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=self.device)
self.query_start_loc = self._make_buffer(self.max_num_reqs + 1, dtype=torch.int32)
self.seq_lens = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
self.num_computed_tokens = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
self.num_scheduled_tokens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:719` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:744`

这些 buffer 是 `ModelRunner` 负责高性能执行的关键。

---

## 6. load_model：ModelRunner 负责真正加载模型

`Worker.load_model()` 只是外层包装，真正加载模型发生在 `GPUModelRunner.load_model()`。

入口：

```python
@instrument(span_name="Loading (GPU)")
def load_model(self, load_dummy_weights: bool = False) -> None:
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5142` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5143`

核心模型加载：

```python
model_loader = get_model_loader(self.load_config)
self.model = model_loader.load_model(
    vllm_config=self.vllm_config, model_config=self.model_config
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5163` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5166`

如果有 LoRA：

```python
if self.lora_config:
    self.model = self.load_lora_model(
        self.model, self.vllm_config, self.device
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5167` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5170`

如果有 speculative decoding drafter：

```python
if hasattr(self, "drafter"):
    logger.info_once("Loading drafter model...")
    if hasattr(self.drafter, "load_model"):
        self.drafter.load_model(self.model)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5171` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5174`

加载后会记录模型内存：

```python
self.model_memory_usage = m.consumed_memory
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5230` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5231`

还会处理：

```text
MoE / EPLB；
communication buffer；
多模态 pruning 能力；
torch.compile；
BreakableCUDAGraphWrapper；
CUDAGraphWrapper；
UBatchWrapper；
offloader post_init。
```

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5200` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5318`

所以：

```text
模型实例化、LoRA 包装、drafter 加载、compile / cudagraph wrapper 都属于 ModelRunner。
```

---

## 7. initialize_kv_cache：ModelRunner 初始化物理 KV cache 和 attention backend

`Worker.initialize_from_config()` 会调用：

```python
self.model_runner.initialize_kv_cache(kv_cache_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:577` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:578`

ModelRunner 侧入口：

```python
def initialize_kv_cache(
    self,
    kv_cache_config: KVCacheConfig,
    is_profiling: bool = False,
) -> None:
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7303` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7307`

它做的事情可以拆成：

```text
1. 保存 kv_cache_config；
2. 添加 encoder-only attention layers；
3. 添加 KV sharing layers；
4. 初始化 attention backend；
5. 初始化 Mamba backend；
6. 准备 kernel block sizes；
7. 创建 attention metadata builders；
8. 根据 KV cache config 重建 InputBatch；
9. 分配 KV cache tensors；
10. 注册 KV transfer group 可访问的 KV cache。
```

对应代码：

```python
self.kv_cache_config = kv_cache_config
self.may_add_encoder_only_layers_to_kv_cache_config()
self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
self.initialize_attn_backend(kv_cache_config, is_profiling=is_profiling)
initialize_mamba_ssu_backend(...)
kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, self.attn_groups)
self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)
self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
kv_caches = self.initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7314` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7340`

如果有 KV transfer group：

```python
kv_transfer_group.register_kv_caches(kv_caches)
kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7351` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7360`

因此，职责边界是：

```text
Scheduler / KVCacheManager：管理逻辑 block 分配；
ModelRunner：管理物理 KV cache tensor、attention backend 和 slot mapping 使用。
```

---

## 8. get_kv_cache_spec：ModelRunner 告诉 EngineCore 需要什么 KV cache

初始化 KV cache 之前，EngineCore 会通过 Executor 向 Worker / ModelRunner 查询 KV cache spec。

ModelRunner 侧：

```python
def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7459`

它会遍历模型里的 attention layers：

```python
attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
for layer_name, attn_module in attn_layers.items():
    ...
    if spec := attn_module.get_kv_cache_spec(self.vllm_config):
        kv_cache_spec[layer_name] = spec
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7470` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7488`

它还会跳过 KV sharing target layer：

```python
if isinstance(attn_module, Attention) and (
    kv_tgt_layer := attn_module.kv_sharing_target_layer_name
):
    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
    continue
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7472` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7484`

所以：

```text
ModelRunner 根据真实模型结构报告 KV cache 需求；
EngineCore 根据这个需求 profile / 分配 KV cache；
Scheduler 根据得到的 KVCacheConfig 做逻辑 block 调度。
```

---

## 9. execute_model 主流程

ModelRunner 的主入口是：

```python
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: IntermediateTensors | None = None,
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4043` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4048`

它的主流程可以概括为：

```text
1. 检查上一次 execute_model_state 是否已被 sample_tokens 消费；
2. 处理 ngram_gpu / KV connector 前置逻辑；
3. _update_states(scheduler_output) 更新请求和 batch 状态；
4. 如果是 EC producer，执行 encoder 并返回空 encoder output；
5. 如果本轮无 token，返回 empty output 或 KV-only output；
6. 准备 num_scheduled_tokens_np / batch size；
7. _prepare_inputs()；
8. 计算 cascade attention prefix；
9. _determine_batch_execution_and_padding()；
10. 处理 Mamba preprocess；
11. _get_slot_mappings()；
12. _build_attention_metadata()；
13. _preprocess()；
14. _model_forward()；
15. pooling 或 compute_logits；
16. 保存 execute_model_state；
17. 返回 None，等待 sample_tokens()。
```

这条链路是 ModelRunner 的核心。

---

## 10. execute_model 开头的状态保护

`execute_model()` 开头会检查：

```python
if self.execute_model_state is not None:
    raise RuntimeError(
        "State error: sample_tokens() must be called "
        "after execute_model() returns None."
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4049` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4053`

这说明：

```text
对于 generation 路径，execute_model() 和 sample_tokens() 是一组配对调用。
```

`execute_model()` 可能先执行 forward / logits，并把采样所需状态保存在：

```python
self.execute_model_state
```

然后必须调用 `sample_tokens()` 消费它。

---

## 11. _update_states：把 SchedulerOutput 合并到 worker 侧状态

`execute_model()` 第一件核心工作是：

```python
deferred_state_corrections_fn = self._update_states(scheduler_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4085` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4086`

`_update_states()` 的注释说明：

```python
"""Update the cached states and the persistent batch with the scheduler
output.

The updated states are used by the `_prepare_inputs` function to create
the input GPU tensors for the model.

The SamplingMetadata is updated and copied to the GPU if there is a
new/resumed/paused/finished request in the batch.
"""
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1127` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1135`

它主要做：

```text
删除 finished requests；
清零新分配 KV block；
释放 encoder cache；
移除本轮未调度的 cached requests；
添加 scheduled_new_reqs；
更新 scheduled_cached_reqs；
更新 block ids；
更新 output token ids；
更新 spec token ids；
添加新请求 / resumed 请求到 InputBatch；
condense batch；
可能重排 batch；
refresh metadata；
返回 async spec decode correction 回调。
```

例如，删除 finished 请求：

```python
for req_id in scheduler_output.finished_req_ids:
    self.requests.pop(req_id, None)
    self.num_prompt_logprobs.pop(req_id, None)
...
for req_id in scheduler_output.finished_req_ids:
    self.input_batch.remove_request(req_id)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1138` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1151`

移除本轮未调度请求：

```python
scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
cached_req_ids = self.input_batch.req_id_to_index.keys()
resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
...
for req_id in unscheduled_req_ids:
    self.input_batch.remove_request(req_id)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1167` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1182`

添加新请求：

```python
req_state = CachedRequestState(
    req_id=req_id,
    prompt_token_ids=new_req_data.prompt_token_ids,
    prompt_embeds=new_req_data.prompt_embeds,
    prompt_is_token_ids=new_req_data.prompt_is_token_ids,
    mm_features=new_req_data.mm_features,
    sampling_params=sampling_params,
    pooling_params=pooling_params,
    generator=generator,
    block_ids=new_req_data.block_ids,
    num_computed_tokens=new_req_data.num_computed_tokens,
    output_token_ids=[],
    lora_request=new_req_data.lora_request,
)
self.requests[req_id] = req_state
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1224` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1238`

更新 cached / running 请求：

```python
for i, req_id in enumerate(req_data.req_ids):
    req_state = self.requests[req_id]
    num_computed_tokens = req_data.num_computed_tokens[i]
    new_block_ids = req_data.new_block_ids[i]
    resumed_from_preemption = req_id in req_data.resumed_req_ids
    num_output_tokens = req_data.num_output_tokens[i]
    req_index = self.input_batch.req_id_to_index.get(req_id)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1284` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1290`

这就是 SchedulerOutput 到 worker-local state 的桥。

---

## 12. SchedulerOutput 在 ModelRunner 眼里是什么

`SchedulerOutput` 定义在 `output.py`：

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: list[int]
    finished_req_ids: set[str]
    free_encoder_mm_hashes: list[str]
    ...
```

位置：`code/vllm/vllm/v1/core/sched/output.py:180` 到 `code/vllm/vllm/v1/core/sched/output.py:245`

ModelRunner 主要消费这些字段：

| 字段 | ModelRunner 用途 |
|---|---|
| `scheduled_new_reqs` | 创建新的 `CachedRequestState` |
| `scheduled_cached_reqs` | 更新已缓存请求的 token / block / output 状态 |
| `num_scheduled_tokens` | 确定每个请求本轮执行多少 token |
| `total_num_scheduled_tokens` | 确定本轮是否需要 forward、batch token 总数 |
| `scheduled_spec_decode_tokens` | 更新 spec token 输入和 spec metadata |
| `scheduled_encoder_inputs` | 执行多模态 encoder / encoder-decoder 输入 |
| `num_common_prefix_blocks` | cascade attention 优化 |
| `finished_req_ids` | 删除 worker 侧缓存状态 |
| `free_encoder_mm_hashes` | 删除 encoder cache |
| `kv_connector_metadata` | KV transfer 处理 |
| `ec_connector_metadata` | encoder cache transfer 处理 |
| `new_block_ids_to_zero` | 清零新分配 KV blocks |

所以对于 ModelRunner 来说：

```text
SchedulerOutput 既是 batch 状态更新指令，
也是本轮 forward 输入构造说明书。
```

---

## 13. _prepare_inputs：准备本轮模型输入

更新完状态后，ModelRunner 会根据当前 `InputBatch` 和 `SchedulerOutput` 准备本轮模型输入。

入口调用：

```python
logits_indices, spec_decode_metadata = self._prepare_inputs(
    scheduler_output,
    num_scheduled_tokens_np,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4128` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4131`

`_prepare_inputs()` 做的事情可以概括为：

```text
1. 根据每个请求的 num_scheduled_tokens 找到本轮 token 范围；
2. 准备 input_ids / positions 所需的索引信息；
3. 计算哪些位置需要 logits；
4. 处理 speculative decoding metadata；
5. 处理多模态 / M-RoPE / XD-RoPE 位置；
6. 为后续 attention metadata 和 preprocess 准备基础张量。
```

这一步还没有真正 forward，它是在把 Scheduler 的“计划”变成 GPU 侧可执行的输入结构。

---

## 14. batch 执行形态：padding、CUDA graph、ubatching

准备输入后，ModelRunner 会决定这一轮 batch 以什么形态执行：

```python
(
    cudagraph_mode,
    batch_desc,
    should_ubatch,
    num_tokens_across_dp,
    cudagraph_stats,
) = self._determine_batch_execution_and_padding(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4143` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4156`

输入包括：

```text
num_tokens_unpadded
num_reqs
num_scheduled_tokens_np
max_num_scheduled_tokens
是否使用 cascade attention
encoder request 数量
```

这一步会决定：

```text
是否走 CUDA graph；
是否 padding 到 graph capture size；
是否使用 ubatching；
DP 之间 token 数如何协调；
返回 batch descriptor。
```

所以 ModelRunner 不只是准备 token，它还负责决定“这一批怎么跑得更快”。

---

## 15. slot mapping 和 attention metadata

ModelRunner 会先准备 slot mapping：

```python
slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
    num_tokens_padded=...,
    num_reqs_padded=...,
    num_tokens_unpadded=num_tokens_unpadded,
    ubatch_slices=ubatch_slices_padded,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4244` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4253`

然后构造 attention metadata：

```python
attn_metadata, spec_decode_common_attn_metadata = (
    self._build_attention_metadata(
        num_tokens=num_tokens_unpadded,
        num_tokens_padded=num_tokens_padded if pad_attn else None,
        num_reqs=num_reqs,
        num_reqs_padded=num_reqs_padded if pad_attn else None,
        max_query_len=max_num_scheduled_tokens,
        ubatch_slices=ubatch_slices_attn,
        logits_indices=logits_indices,
        use_spec_decode=use_spec_decode,
        num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
        cascade_attn_prefix_lens=cascade_attn_prefix_lens,
        slot_mappings=slot_mappings_by_group,
    )
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4255` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4269`

这一步是 attention backend 的关键入口。

可以理解为：

```text
Scheduler 只分配逻辑 block；
ModelRunner 把 block table / slot mapping / query lens / prefix 信息转换成 attention backend 可用的 metadata。
```

---

## 16. _preprocess：生成模型 forward 参数

attention metadata 构造后，ModelRunner 调用：

```python
(
    input_ids,
    inputs_embeds,
    positions,
    intermediate_tensors,
    model_kwargs,
    ec_connector_output,
) = self._preprocess(
    scheduler_output, num_tokens_padded, intermediate_tensors
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4271` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4280`

`_preprocess()` 可以理解为：

```text
把前面准备好的 batch / metadata 进一步变成模型 forward 函数的实参。
```

它输出：

```text
input_ids：本轮 token ids；
inputs_embeds：如果请求使用 prompt embeds 或多模态 embeds；
positions：position ids；
intermediate_tensors：PP 场景上游传来的中间张量；
model_kwargs：模型特定参数；
ec_connector_output：encoder cache connector 输出。
```

到这一步，ModelRunner 才真正具备调用模型的全部输入。

---

## 17. _model_forward：真正模型 forward 在这里发生

真正 forward 调用在：

```python
model_output = self._model_forward(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4320` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4326`

forward 外面包了 forward context：

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

这说明模型 forward 不只是传 `input_ids`，还依赖全局 forward context 中的：

```text
attention metadata；
CUDA graph runtime mode；
batch descriptor；
slot mapping；
ubatch slices；
DP token 数；
```

因此：

```text
ModelRunner 是模型 forward 的真正编排者。
```

---

## 18. forward 后：PP、pooling、logits 三种分支

forward 得到 `model_output` 后，ModelRunner 会进入 postprocess。

### 18.1 非 last PP rank 返回 IntermediateTensors

如果不是 last pipeline rank：

```python
if not get_pp_group().is_last_rank:
    assert isinstance(hidden_states, IntermediateTensors)
    self.kv_connector_output = kv_connector_output
    return hidden_states
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4337` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4343`

这表示当前 rank 只负责模型的一部分层，输出中间张量给下一个 PP stage。

### 18.2 pooling model 返回 pooling output

如果是 pooling model：

```python
if self.is_pooling_model:
    return self._pool(
        hidden_states,
        num_scheduled_tokens,
        num_scheduled_tokens_np,
        kv_connector_output,
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4345` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4352`

这类请求不走 token sampling，而是直接返回 pooling output。

### 18.3 generation model 计算 logits

普通 generation 路径：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4354` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4355`

这里说明：

```text
forward 输出 hidden states；
ModelRunner 根据 logits_indices 取需要采样的位置；
再调用模型的 compute_logits 得到 logits。
```

---

## 19. execute_model_state：为什么 execute_model 经常返回 None

generation 路径下，ModelRunner 不一定在 `execute_model()` 中直接采样。

它会保存临时状态：

```python
self.execute_model_state = ExecuteModelState(
    scheduler_output,
    logits,
    spec_decode_metadata,
    spec_decode_common_attn_metadata,
    hidden_states,
    sample_hidden_states,
    aux_hidden_states,
    ec_connector_output,
    cudagraph_stats,
    slot_mappings,
)
self.kv_connector_output = kv_connector_output
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4386` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4398`

然后：

```python
return None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4405`

为什么？

因为采样前可能需要 Scheduler 提供 grammar bitmask：

```text
execute_model()
  → forward / logits
  → 返回 None
  → Scheduler.get_grammar_bitmask()
  → sample_tokens(grammar_output)
  → ModelRunnerOutput
```

这就是 `execute_model()` 和 `sample_tokens()` 分阶段的原因。

---

## 20. sample_tokens：从 logits 到 ModelRunnerOutput

`sample_tokens()` 入口：

```python
@torch.inference_mode
def sample_tokens(
    self, grammar_output: "GrammarOutput | None"
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4422` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4425`

如果没有 `execute_model_state`，则只返回 KV connector output：

```python
if self.execute_model_state is None:
    kv_connector_output = self.kv_connector_output
    self.kv_connector_output = None
    ...
    return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4426` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4434`

正常 generation 路径会取出 `execute_model_state`：

```python
(
    scheduler_output,
    logits,
    spec_decode_metadata,
    spec_decode_common_attn_metadata,
    hidden_states,
    sample_hidden_states,
    aux_hidden_states,
    ec_connector_output,
    cudagraph_stats,
    slot_mappings,
) = self.execute_model_state
self.execute_model_state = None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4436` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4450`

如果有结构化输出 grammar bitmask：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4452` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4456`

然后采样：

```python
sampler_output = self._sample(logits, spec_decode_metadata)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4458` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4459`

采样后更新状态：

```python
self._update_states_after_model_execute(
    sampler_output.sampled_token_ids, scheduler_output
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4461` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4463`

后续还会处理：

```text
async scheduling 下 PP sampled token 广播；
speculative decoding drafter 提案；
bookkeeping；
logprobs / prompt logprobs；
NaN 统计；
ModelRunnerOutput 构造。
```

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4464` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4589`

---

## 21. ModelRunnerOutput 包含什么

`ModelRunnerOutput` 定义在 `v1/outputs.py`：

```python
@dataclass
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]] = field(default_factory=list)
    logprobs: LogprobsLists | None = None
    prompt_logprobs_dict: dict[str, LogprobsTensors | None] = field(default_factory=dict)
    pooler_output: list[torch.Tensor | None] | None = None
    kv_connector_output: KVConnectorOutput | None = None
    ec_connector_output: ECConnectorOutput | None = None
    num_nans_in_logits: dict[str, int] | None = None
    cudagraph_stats: CUDAGraphStat | None = None
    routed_experts: RoutedExpertsLists | None = None
```

位置：`code/vllm/vllm/v1/outputs.py:231` 到 `code/vllm/vllm/v1/outputs.py:281`

它是 Worker / ModelRunner 返回给 Scheduler 的 batch 级执行结果。

字段含义：

| 字段 | 含义 |
|---|---|
| `req_ids` | 本轮输出涉及的 request id 顺序 |
| `req_id_to_index` | request id 到 batch row 的映射 |
| `sampled_token_ids` | 每个请求本轮生成 token，spec decode 下可能多个 |
| `logprobs` | sampled token logprobs |
| `prompt_logprobs_dict` | prompt logprobs |
| `pooler_output` | pooling / embedding 输出 |
| `kv_connector_output` | KV transfer worker 侧输出 |
| `ec_connector_output` | encoder cache transfer 输出 |
| `num_nans_in_logits` | logits NaN 统计 |
| `cudagraph_stats` | CUDA graph 执行统计 |
| `routed_experts` | MoE routed experts 信息 |

注意：

```text
ModelRunnerOutput 不是用户输出；
Scheduler.update_from_output() 会把它转成 EngineCoreOutputs；
OutputProcessor 才会最终转成 RequestOutput。
```

---

## 22. ModelRunner 和 Scheduler 的职责边界

### Scheduler 负责

```text
维护 waiting / running 队列；
决定本轮哪些请求执行；
决定每个请求调度多少 token；
分配逻辑 KV blocks；
处理 prefix cache / external KV 命中；
处理 preemption；
构造 SchedulerOutput；
用 ModelRunnerOutput 更新请求状态。
```

### ModelRunner 负责

```text
消费 SchedulerOutput；
维护 worker 侧 CachedRequestState；
维护 InputBatch；
把逻辑 block ids 变成 block table / slot mapping；
准备模型输入；
构造 attention metadata；
执行 forward / logits / pooling / sampling；
构造 ModelRunnerOutput。
```

边界一句话：

```text
Scheduler 决定“这轮跑什么”，ModelRunner 决定“这轮怎么跑”。
```

---

## 23. ModelRunner 和 Worker 的职责边界

### Worker 负责

```text
初始化 device；
初始化 distributed；
创建 ModelRunner；
调用 ModelRunner.load_model()；
调用 ModelRunner.initialize_kv_cache()；
调用 ModelRunner.execute_model() / sample_tokens()；
管理 profile、sleep、wake_up、shutdown 等生命周期。
```

### ModelRunner 负责

```text
模型实例和权重；
KV cache tensor；
attention backend；
InputBatch；
请求状态；
forward 输入准备；
模型执行；
sampling；
ModelRunnerOutput。
```

边界一句话：

```text
Worker 是设备侧生命周期壳；ModelRunner 是模型执行内核。
```

---

## 24. ModelRunner 和 Executor 的职责边界

Executor 不直接理解模型输入，也不直接构造 attention metadata。

Executor 负责：

```text
创建 worker；
向所有 worker 广播 execute_model / sample_tokens；
收集输出；
处理 uni / mp / ray / external_launcher 差异。
```

ModelRunner 负责：

```text
单个 worker / rank 上具体如何执行这一轮 SchedulerOutput。
```

所以：

```text
Executor 是调用分发器；
ModelRunner 是被调用后的实际模型执行器。
```

---

## 25. 特殊能力如何插入 ModelRunner 主链路

### 25.1 Structured Output

Structured output 的 grammar bitmask 在 `sample_tokens()` 里应用：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4452` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4456`

这说明结构化输出不是 OutputProcessor 后处理，而是在采样前限制 logits。

### 25.2 Speculative Decoding

spec decode 在初始化时创建 drafter / rejection sampler：

```python
if self.speculative_config and get_pp_group().is_last_rank:
    ...
    self.rejection_sampler = RejectionSampler(
        self.sampler, self.speculative_config, self.device
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:545` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:620`

采样后可能提出 draft tokens：

```python
self._draft_token_ids = self.propose_draft_token_ids(...)
self._copy_draft_token_ids_to_cpu(scheduler_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4481` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4495`

### 25.3 Mamba / hybrid model

Mamba preprocess 在 forward 前：

```python
if self.cache_config.mamba_cache_mode == "align":
    ...
    mamba_utils.preprocess_mamba(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4198` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4216`

Mamba postprocess 在采样后：

```python
if self.cache_config.mamba_cache_mode == "align":
    mamba_utils.postprocess_mamba_align_gpu(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1517` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1537`

### 25.4 KV Connector

execute_model 前处理 preemption / metadata：

```python
if has_kv_transfer_group():
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4075` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4078`

forward 时收集 KV connector output：

```python
self.maybe_get_kv_connector_output(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4315` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4318`

### 25.5 EC Connector / Encoder

如果当前实例是 EC producer：

```python
if has_ec_transfer() and not get_ec_transfer().is_consumer:
    with self.maybe_get_ec_connector_output(...):
        self._execute_mm_encoder(scheduler_output)
        return make_empty_encoder_model_runner_output(scheduler_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4088` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4094`

这表示 encoder cache disaggregation 也插在 ModelRunner 主链路里。

---

## 26. 一个完整例子：普通 generation 请求

假设一个普通文本生成请求本轮被 Scheduler 调度。

ModelRunner 侧链路：

```text
1. _update_states()
   → 添加或更新 CachedRequestState
   → 更新 InputBatch
   → 更新 block ids / num_computed_tokens

2. _prepare_inputs()
   → 准备本轮 input ids / positions 所需信息
   → 计算 logits_indices

3. _build_attention_metadata()
   → 构造 attention backend metadata
   → 使用 block table / slot mapping

4. _preprocess()
   → 生成 input_ids / inputs_embeds / positions / model_kwargs

5. _model_forward()
   → 模型 forward，得到 hidden states

6. compute_logits()
   → 从 sample_hidden_states 计算 logits

7. 保存 execute_model_state
   → execute_model() 返回 None

8. sample_tokens(grammar_output)
   → 可选应用 grammar bitmask
   → _sample()
   → bookkeeping
   → ModelRunnerOutput(sampled_token_ids=...)
```

这就是普通 generation 的完整 ModelRunner 主线。

---

## 27. 一个完整例子：pooling / embedding 请求

如果模型是 pooling model：

```python
self.is_pooling_model = model_config.runner_type == "pooling"
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:450`

forward 后不走 logits / sampling，而是：

```python
if self.is_pooling_model:
    return self._pool(
        hidden_states,
        num_scheduled_tokens,
        num_scheduled_tokens_np,
        kv_connector_output,
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4345` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4352`

因此 pooling 请求链路是：

```text
SchedulerOutput
  → _update_states
  → _prepare_inputs
  → _model_forward
  → _pool
  → ModelRunnerOutput(pooler_output=...)
```

它不会像 generation 那样进入 `sample_tokens()`。

---

## 28. 一个完整例子：Pipeline Parallel 非 last rank

如果启用 PP，当前 rank 不是 last rank，则 forward 后返回中间张量：

```python
if not get_pp_group().is_last_rank:
    assert isinstance(hidden_states, IntermediateTensors)
    self.kv_connector_output = kv_connector_output
    return hidden_states
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4339` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4343`

然后 Worker 会把这些中间张量发给下一个 PP rank。

所以 PP 非 last rank 的 ModelRunner：

```text
只负责自己这一段模型层的 forward；
不计算最终 logits；
不采样 token；
不直接产生最终 ModelRunnerOutput。
```

最终 logits / sampling 通常发生在 last PP rank。

---

## 29. 容易疑惑的点

### 29.1 ModelRunner 是不是 Worker？

不是。

```text
Worker 持有 ModelRunner；
Worker 接收 Executor RPC；
ModelRunner 执行模型细节。
```

### 29.2 ModelRunner 是不是 Scheduler？

不是。

```text
Scheduler 决定 batch 计划；
ModelRunner 执行 batch 计划。
```

### 29.3 execute_model() 为什么经常返回 None？

因为 generation 路径中，forward / logits 和 sampling 被拆开。

```text
execute_model()：forward / logits / 保存 execute_model_state
sample_tokens()：应用 grammar bitmask / sampling / ModelRunnerOutput
```

### 29.4 InputBatch 和 CachedRequestState 有什么区别？

```text
CachedRequestState：单请求长期状态；
InputBatch：当前 persistent batch 的张量化状态。
```

### 29.5 ModelRunnerOutput 是最终用户输出吗？

不是。

```text
ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput
```

### 29.6 KV cache 是 Scheduler 管还是 ModelRunner 管？

两者都参与，但层次不同：

```text
Scheduler / KVCacheManager：逻辑 block 分配；
ModelRunner：物理 KV cache tensor、block table、slot mapping、attention backend 使用。
```

---

## 30. 从“回答问题”的角度总结

如果要问：

```text
ModelRunner 在 vLLM V1 里负责什么？
```

可以回答：

```text
ModelRunner 是 Worker 内部真正执行模型的组件。

它接收 SchedulerOutput，先通过 _update_states() 更新 worker 侧 CachedRequestState 和 InputBatch，
再通过 _prepare_inputs()、_build_attention_metadata()、_preprocess() 构造模型 forward 所需的 input_ids、positions、attention metadata、slot mapping 和 model_kwargs。
随后它调用 _model_forward() 执行模型，
对于 generation 模型计算 logits 并在 sample_tokens() 中采样，
对于 pooling 模型直接生成 pooling output，
最后构造 ModelRunnerOutput 返回给 Scheduler。
```

职责关系可以概括为：

```text
Executor：负责分发执行调用；
Worker：负责设备侧生命周期和控制面；
ModelRunner：负责一次 batch 的模型执行细节；
Scheduler：负责调度决策和状态账本；
OutputProcessor：负责最终用户输出。
```

---

## 31. 最关键流程图

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → Executor.execute_model()
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
      │
      ├─ _update_states()
      │    ├─ finished_req_ids 清理
      │    ├─ scheduled_new_reqs 添加
      │    ├─ scheduled_cached_reqs 更新
      │    ├─ block ids / output ids / spec ids 更新
      │    └─ InputBatch refresh / reorder
      │
      ├─ _prepare_inputs()
      │    ├─ token ranges
      │    ├─ logits_indices
      │    └─ spec_decode_metadata
      │
      ├─ _determine_batch_execution_and_padding()
      │    ├─ CUDA graph mode
      │    ├─ padding
      │    └─ ubatching
      │
      ├─ _get_slot_mappings()
      ├─ _build_attention_metadata()
      ├─ _preprocess()
      │    ├─ input_ids
      │    ├─ inputs_embeds
      │    ├─ positions
      │    └─ model_kwargs
      │
      ├─ _model_forward()
      │
      ├─ if non-last PP rank:
      │    └─ return IntermediateTensors
      │
      ├─ if pooling model:
      │    └─ return ModelRunnerOutput(pooler_output)
      │
      └─ if generation model:
           ├─ compute_logits()
           ├─ save execute_model_state
           └─ return None
                ↓
           GPUModelRunner.sample_tokens(grammar_output)
             ├─ apply_grammar_bitmask()
             ├─ _sample()
             ├─ _update_states_after_model_execute()
             ├─ draft token proposal / bookkeeping
             └─ ModelRunnerOutput
```

---

## 32. 最关键对象关系

```text
SchedulerOutput
  Scheduler 发来的本轮执行计划。

CachedRequestState
  ModelRunner 侧保存的单请求状态。

InputBatch
  ModelRunner 侧当前 batch 的张量化状态。

AttentionMetadata
  attention backend 需要的执行元数据。

ExecuteModelState
  execute_model() 和 sample_tokens() 之间暂存的 logits / hidden states / metadata。

ModelRunnerOutput
  ModelRunner 返回给 Scheduler 的 batch 级真实结果。
```

---

## 33. 和后续专题的关系

本篇回答的是 ModelRunner 的总定位。

后续专题可以继续拆：

```text
04_execute_model_flow.md
  详细串起 EngineCore → Executor → Worker → ModelRunner 的 execute_model 调用链。

05_input_batch_and_state_update.md
  详细解释 _update_states()、CachedRequestState、InputBatch。

06_prepare_inputs_and_attention_metadata.md
  详细解释 token ids、positions、slot mapping、attention metadata。

07_model_forward_and_logits.md
  详细解释 _model_forward()、hidden states、compute_logits、pooling。

08_sampling_and_model_runner_output.md
  详细解释 sample_tokens()、Sampler、grammar、spec decode、ModelRunnerOutput。

09_worker_kv_cache_interaction.md
  详细解释 KV cache tensor、block table、slot mapping、KV connector。
```

最终最小心智模型：

```text
ModelRunner = InputBatch 状态维护 + 输入张量准备 + attention metadata 构造 + 模型 forward + logits/pooling + sampling + ModelRunnerOutput。
```
