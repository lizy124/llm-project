# 09. Attention 和 KV Connector hook 如何衔接？

源码位置：

- `code/vllm/vllm/model_executor\layers\attention\kv_transfer_utils.py`
- `code/vllm/vllm/model_executor\layers\attention\attention.py`
- `code/vllm/vllm/forward_context.py`
- `code/vllm/vllm/v1\worker\kv_connector_model_runner_mixin.py`
- `code/vllm/vllm/v1\worker\gpu_model_runner.py`
- `code/vllm/vllm/distributed\kv_transfer\kv_transfer_state.py`
- `code/vllm/vllm/distributed\kv_transfer\kv_connector\v1\base.py`
- `code/vllm/vllm/v1\outputs.py`

本文用于梳理 KV Connector 为什么挂在 attention layer 边界，`start_load_kv()` / `wait_for_layer_load()` / `save_kv_layer()` / `wait_for_save()` 如何与 `GPUModelRunner.execute_model()`、`set_forward_context()`、`unified_attention_with_output()` 和 backend attention forward 配合。

---

## 1. 本文要回答的问题

```text
KV Connector 在 vLLM V1 中解决什么问题？
KV connector metadata 如何从 SchedulerOutput 进入 worker-side connector？
为什么 start_load_kv() 在 model forward 前调用？
为什么 wait_for_layer_load() 挂在 attention layer entry？
为什么 save_kv_layer() 挂在 attention layer exit？
maybe_transfer_kv_layer 装饰器做了什么？
maybe_get_kv_connector_output() 如何包住整个 forward？
no-forward / 0-token step 如何推进 KV connector？
spec decode 下为什么要 defer finalize？
KVConnectorOutput 如何把 finished / invalid block / stats 回传 Scheduler？
KV connector 如何影响 KV cache layout、CUDA graph 和 attention forward？
```

---

## 2. 一句话回答

KV Connector hook 分两层挂进执行链路：

```text
1. ModelRunner 级别
   maybe_get_kv_connector_output() 包住整个 model forward，负责 bind metadata、start_load_kv、wait_for_save、收集 KVConnectorOutput、clear metadata。

2. Attention layer 级别
   maybe_transfer_kv_layer 包住 unified_attention_with_output()，在每层 attention 入口 wait_for_layer_load(layer_name)，在每层 attention 结束后 save_kv_layer(layer_name, kv_cache, attn_metadata)。
```

它挂在 attention layer 边界的原因是：

```text
进入 attention 前，必须保证当前层需要读取的远端 KV 已经加载到本地 paged KV buffer；
退出 attention 后，当前层新写入或可导出的 KV 已经在本层 kv_cache tensor 中，可以立即启动异步 save。
```

最小链路是：

```text
SchedulerOutput.kv_connector_metadata
  → GPUModelRunner.execute_model()
  → handle_preemptions(...)
  → set_forward_context(...)
  → maybe_get_kv_connector_output(...)
      → bind_connector_metadata(...)
      → start_load_kv(get_forward_context())
      → model forward
          → Attention.forward()
          → unified_attention_with_output()
              → maybe_transfer_kv_layer wrapper
                  → wait_for_layer_load(layer_name)
                  → backend impl.forward(...)
                  → save_kv_layer(layer_name, kv_cache, attn_metadata)
      → wait_for_save()
      → get_finished() / invalid_block_ids / stats / events / worker_meta
      → clear_connector_metadata()
  → ModelRunnerOutput.kv_connector_output
  → Scheduler update_connector_output(...)
```

如果只记住一句话：

```text
ModelRunner 级 hook 管 connector 的“每步生命周期”，attention layer 级 hook 管 connector 的“每层 KV 同步点”。
```

---

## 3. 先区分几类 KV Connector 概念

### 3.1 KV transfer group

全局 KV connector agent 存在于：`code/vllm/vllm/distributed/kv_transfer/kv_transfer_state.py`

关键函数：

```text
get_kv_transfer_group()
has_kv_transfer_group()
is_v1_kv_transfer_group()
ensure_kv_transfer_initialized(...)
ensure_kv_transfer_shutdown()
```

`ensure_kv_transfer_initialized()` 会在配置了 `kv_transfer_config` 且当前实例是 KV transfer instance 时创建 connector：

```text
KVConnectorFactory.create_connector(
  config=vllm_config,
  role=KVConnectorRole.WORKER,
  kv_cache_config=kv_cache_config,
)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_transfer_state.py:72`

所以 worker 侧执行链路中看到的：

```text
get_kv_transfer_group()
```

就是当前 worker 进程的 KV connector agent。

### 3.2 KVConnectorBase_V1

V1 connector 抽象定义在：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:171`

它分成两组接口。

Scheduler-side：

```text
get_num_new_matched_tokens()
update_state_after_alloc()
build_connector_meta()
update_connector_output()
request_finished()
take_events()
```

Worker-side：

```text
bind_connector_metadata()
clear_connector_metadata()
handle_preemptions()
start_load_kv()
wait_for_layer_load()
save_kv_layer()
wait_for_save()
get_finished()
get_block_ids_with_load_errors()
get_kv_connector_stats()
get_kv_connector_kv_cache_events()
build_connector_worker_meta()
```

本文重点是 worker-side 接口如何和 attention forward 接上。

### 3.3 KVConnectorMetadata 和 KVConnectorOutput

`KVConnectorMetadata` 是 Scheduler 侧构造、Worker 侧消费的 metadata。

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:141`

它会通过：

```text
SchedulerOutput.kv_connector_metadata
```

传到 worker。

`KVConnectorOutput` 是 Worker 侧回传 Scheduler 的结果。

位置：`code/vllm/vllm/v1/outputs.py:196`

核心字段：

```text
finished_sending
finished_recving
kv_connector_stats
kv_cache_events
kv_connector_worker_meta
invalid_block_ids
expected_finished_count
```

它最终挂在：

```text
ModelRunnerOutput.kv_connector_output
```

---

## 4. KV connector 生命周期总览

一次有 forward 的 step 中，worker-side KV connector 生命周期是：

```text
GPUModelRunner.execute_model()
  │
  ├─ 如果 has_kv_transfer_group():
  │    └─ get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
  │
  ├─ 准备 input / slot_mapping / attn_metadata
  │
  ├─ with set_forward_context(...):
  │    └─ with maybe_get_kv_connector_output(...):
  │         ├─ bind_connector_metadata(scheduler_output.kv_connector_metadata)
  │         ├─ start_load_kv(get_forward_context())
  │         ├─ yield KVConnectorOutput 给 model forward
  │         │    └─ attention layer 内部：
  │         │         ├─ wait_for_layer_load(layer_name)
  │         │         ├─ backend attention forward
  │         │         └─ save_kv_layer(layer_name, kv_cache, attn_metadata)
  │         ├─ wait_for_save()
  │         ├─ get_finished(...)
  │         ├─ get_block_ids_with_load_errors()
  │         ├─ get stats / events / worker_meta
  │         └─ clear_connector_metadata()
  │
  ├─ postprocess / logits / sample state
  └─ ModelRunnerOutput(kv_connector_output=...)
```

这个生命周期横跨三层：

```text
SchedulerOutput：携带 connector metadata；
ModelRunner：绑定 metadata、启动 load、收集 output；
Attention layer：逐层等待 load、逐层触发 save。
```

---

## 5. KV cache 初始化时先把真实 KV tensor 注册给 connector

KV connector 要能 load/save KV，首先要知道本 worker 上每层 KV cache tensor 在哪里。

`GPUModelRunner.initialize_kv_cache()` 中，在创建 KV cache tensor 后会注册给 connector。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7515`

```text
if has_kv_transfer_group() and not is_profiling:
    kv_transfer_group = get_kv_transfer_group()
    if self.cross_layers_kv_cache is not None:
        kv_transfer_group.register_cross_layers_kv_cache(
            self.cross_layers_kv_cache,
            self.cross_layers_attn_backend,
        )
    else:
        kv_transfer_group.register_kv_caches(kv_caches)
    kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)
```

这一步完成三件事：

```text
1. register_kv_caches(kv_caches)
   connector 获得 layer_name → kv_cache tensor 的映射。

2. register_cross_layers_kv_cache(...)
   如果使用跨层连续 KV cache layout，connector 获得一个包含所有层的统一 KV tensor。

3. set_host_xfer_buffer_ops(copy_kv_blocks)
   connector 获得 host/device block copy 的平台操作。
```

### 5.1 uniform / cross-layer KV cache layout

`KVConnectorModelRunnerMixin.use_uniform_kv_cache()` 判断是否使用跨层统一 KV cache。

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:115`

它要求：

```text
1. 存在 KV transfer group；
2. connector.prefer_cross_layer_blocks 为 True；
3. KV cache config 只有一个 group，且 attention group 也只有一个；
4. KV cache spec 是 AttentionSpec；
5. backend 支持带 num_layers 维度的 stride order。
```

为什么 connector 会偏好 cross-layer blocks？

```text
如果同一个 block number 下所有 layer 的 KV 数据在内存中更连续，
connector 可以更高效地一次传输某个 block 的跨层 KV。
```

但这个布局必须同时被 attention backend 支持，否则 attention kernel 无法正确解释 KV cache tensor。

---

## 6. execute_model 开头：先处理 preemption / eviction

`GPUModelRunner.execute_model()` 开头有：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4128`

```text
if has_kv_transfer_group():
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
```

`handle_preemptions()` 的接口说明在：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:285`

它用于：

```text
在 preempted requests 或 evicted blocks 被覆盖之前处理它们。
```

典型原因：

```text
有些 connector 会异步保存被抢占或被驱逐的 KV；
如果等 block 被新请求覆盖后再处理，就来不及拿到旧 KV 内容。
```

因此这一步必须在：

```text
_update_states()
_prepare_inputs()
attention forward
```

之前发生。

---

## 7. maybe_get_kv_connector_output() 包住整个 forward

`KVConnectorModelRunnerMixin` 定义在：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:33`

入口：

```text
maybe_get_kv_connector_output(scheduler_output, defer_finalize=False)
```

位置：`kv_connector_model_runner_mixin.py:50`

如果没有 KV transfer group，它返回 `nullcontext()`。

如果有 KV transfer group，它进入：

```text
_get_kv_connector_output(...)
```

### 7.1 _get_kv_connector_output() 的进入阶段

位置：`kv_connector_model_runner_mixin.py:77`

进入 context 时：

```text
output = KVConnectorOutput()
kv_connector = get_kv_transfer_group()
kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
kv_connector.start_load_kv(get_forward_context())
yield output
```

这里有两个关键点。

第一，metadata 每个 execute step 都重新绑定：

```text
bind_connector_metadata(...)
```

它告诉 worker-side connector：

```text
本轮哪些请求需要 load；
哪些请求需要 save；
哪些 block / layer / token 范围参与 transfer；
哪些异步状态需要推进。
```

第二，`start_load_kv()` 接收当前 `ForwardContext`：

```text
start_load_kv(get_forward_context())
```

这意味着 connector 可以看到：

```text
attn_metadata
slot_mapping
batch_descriptor
ubatch_slices
cudagraph_runtime_mode
no_compile_layers
```

从而按当前 forward 的真实 attention layout 启动 KV load。

### 7.2 为什么 start_load_kv() 在 model forward 前

`start_load_kv()` 的接口说明：

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:292`

```text
Start loading the KV cache from the connector to vLLM's paged KV buffer.
This is called from the forward context before the forward pass to enable async loading during model execution.
```

目的不是马上阻塞到所有层加载完，而是：

```text
尽早启动异步 load；
让远端 KV copy 和模型前几层计算尽量重叠；
后续到某一层 attention 前，再精确 wait 这一层。
```

所以 V1 connector 的设计是 layer-by-layer pipelining：

```text
start_load_kv()：启动整轮或多层异步 load；
wait_for_layer_load(layer_name)：只在当前层 attention 入口阻塞到本层可用。
```

### 7.3 _get_kv_connector_output() 的退出阶段

context 退出时：

```text
if wait_for_save and not defer_finalize:
    kv_connector.wait_for_save()

output.finished_sending, output.finished_recving = kv_connector.get_finished(...)
output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
output.kv_connector_stats = kv_connector.get_kv_connector_stats()
output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()
output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()

if not defer_finalize:
    kv_connector.clear_connector_metadata()
```

位置：`kv_connector_model_runner_mixin.py:99`

这说明 `KVConnectorOutput` 是在 forward context 退出时被填充的。

---

## 8. maybe_transfer_kv_layer 挂在 attention layer 边界

文件：`code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py`

核心装饰器：

```text
maybe_transfer_kv_layer(func)
```

它被用于：

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:811`

```python
@eager_break_during_capture
@maybe_transfer_kv_layer
def unified_attention_with_output(...):
    ...
```

也就是说，所有走 `unified_attention_with_output()` 的标准 attention backend 都会经过这个 hook。

### 8.1 装饰器什么时候 no-op

进入 wrapper 后先判断：

```text
if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
    return func(*args, **kwargs)
```

位置：`kv_transfer_utils.py:39`

所以它只对 V1 KV connector 生效。

随后它取出 `layer_name`，并调用：

```text
attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
connector = get_kv_transfer_group()
```

如果：

```text
attn_metadata is None
或 connector.has_connector_metadata() 为 False
```

也会直接执行原函数。

这保证 profile / dummy run / no metadata 场景不会误触发 KV transfer。

### 8.2 attention entry：wait_for_layer_load(layer_name)

如果 connector 可用且 metadata 已绑定：

```text
connector.wait_for_layer_load(layer_name)
```

位置：`kv_transfer_utils.py:51`

它发生在真正 attention forward 之前。

为什么是 attention entry？

```text
attention kernel 会读取当前层历史 KV；
如果这些 KV 来自远端或外部 cache，必须在 kernel 读取前确保已经加载到本地 paged KV buffer；
layer_name 可以让 connector 只等待当前层，而不是等待所有层。
```

这给了 connector 做流水线的空间：

```text
第 i 层 attention 等第 i 层 KV；
第 i+1 层 KV 可以仍在异步加载；
模型计算和 KV transfer 有机会重叠。
```

### 8.3 attention exit：save_kv_layer(layer_name, kv_cache, attn_metadata)

attention 执行后：

```text
connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

位置：`kv_transfer_utils.py:57`

为什么是 attention exit？

```text
当前层 attention forward 已经完成当前 token 的 KV cache update；
kv_cache tensor 中有这一层最新可保存的 KV；
attn_metadata 告诉 connector 本轮哪些 request / token / block 是有效的；
此时可以启动异步 save，而不用等整个模型 forward 结束。
```

这同样支持 layer-by-layer save pipelining：

```text
第 i 层 attention 结束后，第 i 层 KV 可以开始保存；
模型继续跑第 i+1 层；
save 和后续计算重叠。
```

---

## 9. unified_attention_with_output 是 hook 的挂载点

`unified_attention_with_output()` 位于：`code/vllm/vllm/model_executor/layers/attention/attention.py:811`

它的主逻辑是：

```text
1. 根据 layer_name 调 get_attention_context(layer_name)；
2. 得到 attn_metadata、self、kv_cache；
3. 调 self.impl.forward(..., kv_cache, attn_metadata, output=output)。
```

位置：`attention.py:827`

因为这里同时具备三类信息：

```text
layer_name：知道当前是哪一层；
kv_cache：当前层真实 paged KV buffer；
attn_metadata：本轮有效 request / block / slot / seq lens 信息。
```

所以这是 connector hook 的理想位置。

如果 hook 挂得更早，例如模型 forward 外层：

```text
只能看到整轮 batch，不知道当前 attention layer 的 precise execution point。
```

如果 hook 挂得更晚，例如 backend kernel 内部：

```text
需要每个 backend 都实现 connector 逻辑，无法统一。
```

挂在 `unified_attention_with_output()` 的好处是：

```text
对上统一所有 attention layer；
对下不侵入 backend impl；
还能精确围住每层 attention kernel。
```

---

## 10. KV connector metadata 如何进入 ForwardContext

严格说，connector metadata 本身不存进 `ForwardContext`。

流程是：

```text
SchedulerOutput.kv_connector_metadata
  → kv_connector.bind_connector_metadata(...)
  → connector 内部保存 _connector_metadata
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:211`

`ForwardContext` 保存的是 attention 执行所需的 batch runtime 信息：

```text
attn_metadata
slot_mapping
batch_descriptor
ubatch_slices
cudagraph_runtime_mode
no_compile_layers
```

位置：`code/vllm/vllm/forward_context.py:128`

connector 通过 `start_load_kv(get_forward_context())` 读取这些 runtime 信息。

所以更准确的关系是：

```text
connector metadata 进入 connector；
attention metadata 进入 ForwardContext；
connector 在 start_load_kv() 中同时使用两者。
```

### 10.1 has_connector_metadata()

attention hook 中会检查：

```text
connector.has_connector_metadata()
```

位置：`kv_transfer_utils.py:47`

它对应：

```text
self._connector_metadata is not None
```

位置：`base.py:243`

这保证只有在 `_get_kv_connector_output()` 已经 bind metadata 的 forward 作用域内，layer hook 才真正执行 load/save。

---

## 11. no-forward / 0-token step 如何处理

如果本轮没有 scheduled tokens：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4149`

```text
if not num_scheduled_tokens:
    if not has_kv_transfer_group():
        return EMPTY_MODEL_RUNNER_OUTPUT
    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
```

这说明：

```text
没有模型 forward，不代表没有 KV connector 工作。
```

### 11.1 kv_connector_no_forward()

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:35`

它做：

```text
with set_forward_context(None, vllm_config),
     _get_kv_connector_output(scheduler_output, wait_for_save=False):
    pass

return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
```

这条路径的意义是：

```text
即使本轮没有 token 需要模型计算，connector 仍然可以推进异步 load / recv / finished 状态；
Scheduler 仍然需要知道 finished_recving、invalid_block_ids、worker_meta 等信息。
```

### 11.2 为什么 wait_for_save=False

no-forward 场景没有新的 attention layer 执行，也就通常没有本轮新触发的 per-layer `save_kv_layer()`。

因此它传：

```text
wait_for_save=False
```

避免无意义等待。

但 `_get_kv_connector_output()` 仍然会收集：

```text
finished_sending
finished_recving
invalid_block_ids
stats
events
worker_meta
```

---

## 12. spec decode 下为什么 defer finalize

在 `GPUModelRunner.execute_model()` 中：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4354`

```text
defer_kv_connector_finalize = self.speculative_config is not None
```

然后传给：

```text
maybe_get_kv_connector_output(..., defer_finalize=defer_kv_connector_finalize)
```

如果 `defer_finalize=True`，`_get_kv_connector_output()` 退出时不会：

```text
wait_for_save()
clear_connector_metadata()
```

对应逻辑在：`kv_connector_model_runner_mixin.py:99`

原因写在注释里：

```text
When spec decode is enabled, defer connector finalization
(wait_for_save + clear metadata) until after draft model runs.
```

也就是说：

```text
target model forward 后，draft model 可能还要继续使用同一 connector metadata 保存自己的 KV；
如果此时清掉 metadata，draft model 的 KV save/load 就失去上下文；
所以 finalize 延后到 sample_tokens() 中 draft model 相关逻辑之后。
```

最终在 `sample_tokens()` 后段：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4687`

```text
if spec_config is not None:
    self.finalize_kv_connector()
```

`finalize_kv_connector()` 做：

```text
kv_connector.wait_for_save()
kv_connector.clear_connector_metadata()
```

位置：`kv_connector_model_runner_mixin.py:64`

---

## 13. KVConnectorOutput 如何回到 Scheduler

`_get_kv_connector_output()` 会填充 `KVConnectorOutput`。

字段来源：

```text
finished_sending, finished_recving
  ← kv_connector.get_finished(scheduler_output.finished_req_ids)

invalid_block_ids
  ← kv_connector.get_block_ids_with_load_errors()

kv_connector_stats
  ← kv_connector.get_kv_connector_stats()

kv_cache_events
  ← kv_connector.get_kv_connector_kv_cache_events()

kv_connector_worker_meta
  ← kv_connector.build_connector_worker_meta()
```

位置：`kv_connector_model_runner_mixin.py:102`

随后在 generation 路径中，`GPUModelRunner` 把它保存到：

```text
self.kv_connector_output
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4402`

在 `sample_tokens()` 构造 `ModelRunnerOutput` 时：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4693`

```text
kv_connector_output = self.kv_connector_output
self.kv_connector_output = None
...
ModelRunnerOutput(kv_connector_output=kv_connector_output)
```

如果是 no-forward 路径，则直接：

```text
ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
```

### 13.1 finished_sending / finished_recving

这两个字段用于告诉 Scheduler：

```text
哪些请求的异步 send/save 已完成；
哪些请求的异步 recv/load 已完成。
```

Scheduler 侧 connector 可以用它更新等待状态，例如：

```text
WAITING_FOR_REMOTE_KVS → 可以继续调度；
异步 saving 完成 → block 可以真正释放或状态可以收尾。
```

### 13.2 invalid_block_ids

`invalid_block_ids` 表示外部计算或外部加载的 KV block 出现 load error。

接口说明在：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:375`

语义：

```text
引用这些 block 的请求应被重新调度为本地 recompute，
不能继续相信这些外部 KV。
```

它需要和 `get_finished()` 配合：

```text
即使 load 失败，请求也必须通过 finished_recving 通知完成这次异步加载流程；
失败的 block ids 最迟要在同一 pass 出现在 invalid_block_ids。
```

### 13.3 stats / events / worker_meta

这些字段用于：

```text
kv_connector_stats：connector 性能 / 传输统计；
kv_cache_events：KV cache 事件；
kv_connector_worker_meta：worker → scheduler 的 connector 自定义元信息。
```

`KVConnectorWorkerMetadata` 支持跨 worker 聚合：

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:150`

```text
aggregate(other)
```

---

## 14. attention hook 和 backend KV update 的关系

attention hook 包在 `unified_attention_with_output()` 外层。

标准 attention path：

```text
Attention.forward()
  → optional unified_kv_cache_update()
  → unified_attention_with_output()
      → maybe_transfer_kv_layer
          → wait_for_layer_load()
          → impl.forward()
          → save_kv_layer()
```

这里要注意两种 KV update 模式。

### 14.1 backend forward 内更新 KV cache

抽象基类默认：

```text
forward_includes_kv_cache_update=True
```

此时 KV 写入发生在：

```text
impl.forward()
```

`save_kv_layer()` 在 `impl.forward()` 之后调用，因此能看到更新后的 KV cache。

但当前很多 V1 backend 会显式覆盖为 `False`，例如 FlashAttention、FlashInfer、Triton、Flex、CPU、ROCm、TurboQuant 等。追踪这些 backend 时，应按 separate KV update 路径理解。

### 14.2 separate KV update backend

常见 V1 backend：

```text
forward_includes_kv_cache_update=False
```

此时 `Attention.forward()` 会先调用：

```text
unified_kv_cache_update(key, value, layer_name)
```

再调用 `unified_attention_with_output()`。

因此 hook 顺序变成：

```text
unified_kv_cache_update()
  → KV 已写入 paged buffer
unified_attention_with_output()
  → wait_for_layer_load()
  → impl.forward()
  → save_kv_layer()
```

这看起来像是先写 KV，再 wait load。

为什么仍然合理？

```text
unified_kv_cache_update 写的是本轮当前 token 的 K/V；
wait_for_layer_load 等的是 connector 需要加载到本地的远端 / 外部历史 KV；
attention kernel 执行前，两者都必须就位。
```

同时 `kv_cache_dummy_dep` 会保证编译器不把 attention forward 重排到 KV update 前面。

---

## 15. 为什么 hook 不直接放进 backend impl

如果把 connector hook 写进每个 backend impl，会出现几个问题：

```text
1. 每个 backend 都要重复实现 connector wait/save；
2. FlashAttention / FlashInfer / Triton / MLA / CPU / wrapper backend 容易行为不一致；
3. KV connector 逻辑会侵入 kernel backend；
4. 新 backend 必须理解外部 KV transfer 细节；
5. cross attention / encoder-only / chunked local 等 wrapper 更难统一。
```

挂在 `unified_attention_with_output()` 的好处是：

```text
1. 所有标准 attention backend 统一经过；
2. 能拿到 layer_name、kv_cache、attn_metadata；
3. 不需要修改 backend impl；
4. 可以精确包住每层 attention 执行边界；
5. 可以利用 ForwardContext 中统一的 layer registry。
```

一句话：

```text
unified_attention_with_output 是 attention backend 的统一门面，也是 KV connector layer hook 的最佳挂载点。
```

---

## 16. CUDA graph / compile 对 KV connector hook 的影响

KV connector 的 layer-by-layer load/save 是 Python / runtime 侧同步点，不适合被完整捕获进 full CUDA graph。

`KVConnectorBase_V1` 提供：

```text
requires_piecewise_for_cudagraph(extra_config)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:609`

接口说明：

```text
Connectors that use asynchronous layer-by-layer operations
(wait_for_layer_load/save_kv_layer) should override this method
to return True when those operations are enabled.
These operations cannot be captured in CUDA graphs and will be skipped during replay,
causing data races.
PIECEWISE mode allows Python code to execute between graph pieces,
ensuring proper synchronization.
```

也就是说：

```text
如果 connector 依赖每层 wait/save，full graph replay 可能跳过 Python hook；
这会导致 attention kernel 读取尚未 load 完的 KV，或 KV 被覆盖前 save 未完成；
因此这类 connector 需要 PIECEWISE CUDA graph，让 Python hook 能在 graph pieces 之间执行。
```

`unified_attention_with_output()` 上还有：

```text
@eager_break_during_capture
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:811`

这也体现了 attention 边界是 compile / graph 的特殊边界。

---

## 17. KV connector 和 KV cache layout 的关系

KV connector 不只是运行时 hook，也会影响 KV cache layout。

### 17.1 connector 可以要求特定 KV cache layout

`KVConnectorBase_V1` 提供：

```text
get_required_kvcache_layout(vllm_config)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:584`

一些 connector 会 override 它，例如 NIXL、LMCache、Mooncake 等。

这类要求通常与外部 KV 传输的数据布局有关：

```text
外部系统希望 HND 或 NHD；
connector 需要更容易按 block / layer / head 维度 copy；
attention backend 也必须能解释对应 layout。
```

### 17.2 connector 可以偏好 cross-layer blocks

`prefer_cross_layer_blocks` 默认是 False。

位置：`base.py:176`

如果 connector 返回 True，ModelRunner 可能尝试使用 cross-layer KV cache：

```text
register_cross_layers_kv_cache(kv_cache, attn_backend)
```

这样 connector 可以一次处理跨层连续 block。

但是否可用取决于：

```text
模型是否只有统一 attention group；
backend 是否支持带 num_layers 维度的 stride order；
KV cache spec 是否是 AttentionSpec；
connector 是否明确 prefer_cross_layer_blocks。
```

---

## 18. async load / WAITING_FOR_REMOTE_KVS 的执行含义

Scheduler-side connector 可能通过：

```text
get_num_new_matched_tokens(request, num_computed_tokens)
```

告诉 Scheduler：

```text
外部 KV cache 中有多少 token 可复用；
这些 token 是否会异步加载。
```

位置：`base.py:453`

如果返回 async load，Scheduler 可能让请求进入类似：

```text
WAITING_FOR_REMOTE_KVS
```

的等待状态，并通过后续 `SchedulerOutput.kv_connector_metadata` 通知 worker 继续推进 load。

Worker 侧对应逻辑是：

```text
有 token：正常 forward，start_load_kv + layer wait/save；
无 token：kv_connector_no_forward()，只推进 connector output；
load 完成：finished_recving 回传 Scheduler；
load 失败：invalid_block_ids 回传 Scheduler，触发 recompute。
```

所以 KV connector 不是普通“forward 后处理”，它会参与调度状态机。

---

## 19. 与 attention metadata 的关系

`save_kv_layer(layer_name, kv_cache, attn_metadata)` 接收的是 backend-specific `AttentionMetadata`。

这很重要，因为 connector 保存 KV 时需要知道：

```text
本轮哪些 request 有效；
每个 request 的 seq lens / query lens；
block table 和 slot mapping 如何解释；
哪些 token 是 prefill / decode / spec decode；
哪些 padded token 不应保存；
cascade / DCP / cross attention 等特殊路径如何影响 KV 范围。
```

不同 backend 的 metadata 结构不同，但 connector 接口统一拿到：

```text
AttentionMetadata
```

具体 connector 可以根据自己支持的 backend / metadata 类型解释它。

### 19.1 attn_metadata is None 时跳过

`maybe_transfer_kv_layer` 中：

```text
if attn_metadata is None or not connector.has_connector_metadata():
    return func(*args, **kwargs)
```

位置：`kv_transfer_utils.py:47`

这避免了 profile run / dummy run / no metadata path 中误触发 KV load/save。

---

## 20. 一个完整例子：远端 KV load + 普通 decode

假设：

```text
请求命中外部 KV cache；
Scheduler 已经分配本地 blocks 用于接收远端 KV；
本轮需要执行 decode；
connector 支持异步 layer-by-layer load。
```

链路：

```text
1. Scheduler 构造 SchedulerOutput.kv_connector_metadata。

2. GPUModelRunner.execute_model() 开头：
   handle_preemptions(kv_connector_metadata)。

3. ModelRunner 准备 input、slot_mapping、attn_metadata。

4. set_forward_context(attn_metadata, slot_mapping, ...)。

5. maybe_get_kv_connector_output 进入：
   bind_connector_metadata(kv_connector_metadata)
   start_load_kv(get_forward_context())

6. model forward 到第 i 层 attention：
   unified_attention_with_output(...)
   → maybe_transfer_kv_layer
   → wait_for_layer_load(layer_i)

7. 第 i 层 backend impl.forward：
   读取本地 paged KV，其中包含已 load 的远端 KV；
   写入本轮新 token KV；
   计算 attention output。

8. 第 i 层 attention exit：
   save_kv_layer(layer_i, kv_cache, attn_metadata)
   启动本层 KV save / export。

9. 所有层 forward 后：
   wait_for_save()
   get_finished(...)
   get_block_ids_with_load_errors()
   clear_connector_metadata()

10. ModelRunnerOutput.kv_connector_output 回到 Scheduler。
```

---

## 21. 一个完整例子：0-token async load 推进

假设某些请求正在等待外部 KV 异步 load，本轮没有 token 需要模型 forward。

链路：

```text
1. GPUModelRunner.execute_model()
   num_scheduled_tokens = 0。

2. 如果没有 KV connector：
   返回 EMPTY_MODEL_RUNNER_OUTPUT。

3. 如果有 KV connector：
   kv_connector_no_forward(scheduler_output, vllm_config)。

4. kv_connector_no_forward 内部：
   set_forward_context(None, vllm_config)
   _get_kv_connector_output(..., wait_for_save=False)

5. connector 可以推进 start_load_kv / get_finished / invalid_block_ids / stats。

6. 返回 ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)。
```

这说明：

```text
KV connector 可以让 engine 在没有模型 token 的 step 中仍然推进外部 KV 状态。
```

---

## 22. 一个完整例子：spec decode defer finalize

假设启用了 speculative decoding。

链路：

```text
1. target model forward：
   maybe_get_kv_connector_output(..., defer_finalize=True)

2. context 退出时：
   收集 KVConnectorOutput；
   不 wait_for_save；
   不 clear_connector_metadata。

3. draft model / proposer 继续运行，可能也触发 KV save。

4. sample_tokens() 后段：
   finalize_kv_connector()
     → wait_for_save()
     → clear_connector_metadata()
```

这样保证：

```text
target model 和 draft model 的 KV connector 操作共享同一轮 connector metadata 生命周期，
不会在 draft model 运行前过早清理 metadata。
```

---

## 23. 容易疑惑的点

### 23.1 KV connector metadata 是放在 ForwardContext 里吗？

不是。

```text
KVConnectorMetadata 绑定到 connector 自己内部；
ForwardContext 保存 attention metadata / slot mapping / batch runtime；
connector.start_load_kv(get_forward_context()) 同时利用两者。
```

### 23.2 为什么 wait_for_layer_load 不在 model forward 前一次性等待？

因为一次性等待会损失流水线机会。

```text
layer-by-layer wait 可以让后续层 KV load 与前面层计算重叠。
```

### 23.3 save_kv_layer 保存的是哪个 KV？

它拿到的是当前 attention layer 的 `kv_cache` tensor 和当前层 `attn_metadata`。

connector 根据 metadata 判断应该保存哪些 request / block / token 范围。

### 23.4 no-forward step 为什么还需要 connector？

因为异步 KV load / recv / send 的状态可能仍在推进，Scheduler 需要收到 `finished_recving`、`finished_sending` 或 `invalid_block_ids`。

### 23.5 KV connector hook 会影响所有 backend 吗？

只要 backend 走标准 `unified_attention_with_output()`，就会经过 `maybe_transfer_kv_layer`。

如果某个特殊路径绕过该统一入口，就需要自己保证 KV connector 语义。

### 23.6 attn_metadata 为 None 时为什么不 hook？

profile / dummy run 中可能没有真实 metadata，也不应该触发真实 KV transfer。

### 23.7 connector 为什么要知道 attention metadata？

因为 KV cache tensor 只是物理内存，metadata 才说明本轮哪些 slot / block / request 是有效 KV。

### 23.8 full CUDA graph 为什么可能和 connector 冲突？

因为 layer-by-layer `wait_for_layer_load()` / `save_kv_layer()` 是 Python runtime 同步点，如果被 full graph replay 跳过，就可能出现数据 race。

这类 connector 应通过 `requires_piecewise_for_cudagraph()` 要求 PIECEWISE 模式。

---

## 24. 最终可以记成一张表

| 阶段 | 主要函数 / 类 | 核心输入 | 核心输出 | 作用 |
|---|---|---|---|---|
| connector 初始化 | `ensure_kv_transfer_initialized()` | `kv_transfer_config`、`kv_cache_config` | worker-side connector | 创建全局 KV transfer group |
| KV tensor 注册 | `initialize_kv_cache()` | `kv_caches` / `cross_layers_kv_cache` | connector 内部 KV 引用 | 让 connector 知道每层 KV buffer |
| preemption 处理 | `handle_preemptions()` | `kv_connector_metadata` | connector 内部状态 | block 覆盖前处理 evicted/preempted KV |
| forward context | `set_forward_context()` | `attn_metadata`、slot_mapping | `ForwardContext` | 给 connector 和 attention layer 提供 runtime batch 信息 |
| 绑定 metadata | `bind_connector_metadata()` | `SchedulerOutput.kv_connector_metadata` | connector `_connector_metadata` | 设置本轮 connector 指令 |
| 启动 load | `start_load_kv()` | `ForwardContext` | 异步 load work | 在 model forward 前启动 KV load |
| 层入口等待 | `wait_for_layer_load()` | `layer_name` | 当前层 KV ready | attention kernel 读 KV 前同步 |
| attention 执行 | `impl.forward()` | Q/K/V、KV cache、metadata | attention output | 真实读写 paged KV 并计算 attention |
| 层退出保存 | `save_kv_layer()` | `layer_name`、`kv_cache`、`attn_metadata` | 异步 save work | 当前层 KV 可导出时启动保存 |
| forward 退出 | `wait_for_save()` | pending saves | save 完成 | 防止 KV buffer 被覆盖前 save 未完成 |
| 输出收集 | `get_finished()` 等 | finished req ids | `KVConnectorOutput` | 回传 finished / invalid / stats / events |
| 清理 metadata | `clear_connector_metadata()` | connector state | None | 结束本轮 connector 生命周期 |

---

## 25. 总结

Attention 和 KV Connector 的衔接可以压缩成下面这条线：

```text
Scheduler builds KVConnectorMetadata
  → ModelRunner binds metadata
  → ModelRunner starts async KV load before forward
  → attention layer waits for current layer KV before kernel
  → backend attention forward reads/writes paged KV cache
  → attention layer starts saving current layer KV after kernel
  → ModelRunner waits save and collects KVConnectorOutput
  → Scheduler consumes finished / invalid / stats / worker_meta
```

职责边界是：

```text
Scheduler-side connector：决定外部 KV 命中、构造 metadata、消费 worker output；
ModelRunner connector mixin：管理每个 execute step 的 connector 生命周期；
attention hook：提供每层 load/save 同步点；
backend impl：仍然只负责解释 metadata、读写 KV cache、执行 attention kernel。
```

如果只记住最小心智模型：

```text
KV Connector 不直接替代 attention backend；它通过 ModelRunner 级生命周期 hook 和 attention layer 级 wait/save hook，把外部 KV 的异步 load/save 精确插入到每层 attention 读写 paged KV cache 的边界上。
```