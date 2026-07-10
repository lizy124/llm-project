# 07. Worker / ModelRunner 如何消费 KV Connector metadata？

源码位置：

- `code/vllm/vllm/distributed/kv_transfer/kv_transfer_state.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/factory.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/utils.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py`
- `code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`

本文承接 `04_scheduler_kv_connector_flow.md`：Scheduler 侧已经把外部 KV 的 load / save / preemption / block mapping 计划封装进 `SchedulerOutput.kv_connector_metadata`。本篇只看 Worker / ModelRunner 侧如何消费这个 metadata，如何把外部 KV 真正 load 到本地 paged KV cache，如何在 attention 层把本地 KV save 出去，以及如何把完成状态通过 `ModelRunnerOutput.kv_connector_output` 回传给 Scheduler。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的写法，本篇按“先定 Worker 侧角色，再走 execute_model 主链路，再拆 load / save / output / 聚合”的顺序组织。

要回答的问题分成 12 组：

```text
1. Worker 侧 KV Connector 是哪一层？和 Scheduler 侧 connector 的边界是什么？
2. Worker 什么时候创建 role=WORKER 的 connector？
3. KV cache tensor 如何注册给 Worker connector？
4. SchedulerOutput.kv_connector_metadata 如何进入 GPUModelRunner.execute_model()？
5. ModelRunner 为什么要在 forward context 中包一层 maybe_get_kv_connector_output()？
6. start_load_kv()、wait_for_layer_load()、save_kv_layer()、wait_for_save() 的时序是什么？
7. 没有 scheduled tokens 时为什么仍然可能执行 KV connector？
8. KVConnectorOutput 里 finished_recving / finished_sending / invalid_block_ids / worker_meta 分别表示什么？
9. 多 worker 输出如何被 KVOutputAggregator 聚合？
10. ExampleConnector 如何展示最小 load / save 逻辑？
11. MooncakeStore / KVPool 类 connector 在 Worker 侧多了哪些异步线程和外部存储细节？
12. Worker 侧输出如何闭环回 Scheduler？
```

阅读顺序建议：

```text
04_scheduler_kv_connector_flow.md
  → 05_external_kv_load_flow.md
  → 07_worker_kv_connector_flow.md
  → 06_external_kv_save_flow.md
  → 08_invalid_blocks_and_recompute.md
  → 09_deferred_free_and_async_safety.md
```

本篇重点讲 Worker / ModelRunner 侧，不重复展开 Scheduler 如何查询外部命中、如何分配 blocks；这些已经在 `04_scheduler_kv_connector_flow.md` 中梳理。

---

## 1. 一句话回答

Worker / ModelRunner 侧 KV Connector 负责把 `SchedulerOutput.kv_connector_metadata` 变成真实的 KV cache 数据搬运，并把搬运完成、失败、统计和 worker-side metadata 汇总成 `ModelRunnerOutput.kv_connector_output` 返回 Scheduler。

它负责：

```text
1. 在 worker 初始化阶段创建 role=WORKER 的 connector；
2. 在 KV cache 初始化完成后，把本地 paged KV cache tensor 注册给 connector；
3. 每轮 execute_model() 前绑定 Scheduler 下发的 kv_connector_metadata；
4. 处理 preemption / eviction 相关的异步保存保护；
5. 在 forward 前启动 KV load；
6. 在每个 attention layer 执行前等待该层 KV load 完成；
7. 在每个 attention layer 执行后触发该层 KV save；
8. 在 forward 结束后等待必要的 save 完成；
9. 收集 finished_recving / finished_sending / invalid_block_ids / stats / events / worker_meta；
10. 清理本轮 metadata，避免污染下一轮执行。
```

它不负责：

```text
1. 不负责决定外部 KV 命中多少 token；
2. 不负责决定本轮是否 load_kv_async；
3. 不负责本地 block 分配；
4. 不负责 waiting / running / preempted 队列状态迁移；
5. 不负责 Scheduler 中的 recompute / fail 策略决策。
```

一句话记忆：

```text
Scheduler 侧 connector 负责“计划 KV transfer”，Worker 侧 connector 负责“执行 KV transfer”。
```

---

## 2. 一句话总览链路

Worker 侧最小主链路是：

```text
GPUWorker.initialize_from_config(kv_cache_config)
  → ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
  → KVConnectorFactory.create_connector(role=WORKER)
  → GPUModelRunner.initialize_kv_cache(kv_cache_config)
  → register_kv_caches(...) / register_cross_layers_kv_cache(...)

EngineCore.step()
  → Executor.execute_model(scheduler_output)
  → Worker.execute_model(scheduler_output)
  → GPUModelRunner.execute_model(scheduler_output)
  → handle_preemptions(kv_connector_metadata)
  → _update_states() / _prepare_inputs() / _build_attention_metadata()
  → set_forward_context(attn_metadata, ...)
  → legacy: maybe_get_kv_connector_output(scheduler_output)
    或新版: kv_connector.pre_forward(scheduler_output)
      → bind_connector_metadata(kv_connector_metadata)
      → start_load_kv(forward_context)
      → model forward
          → attention layer entry: wait_for_layer_load(layer_name)
          → attention layer exit: save_kv_layer(layer_name, kv_cache, attn_metadata)
      → wait_for_save()
      → get_finished(finished_req_ids)
      → get_block_ids_with_load_errors()
      → build_connector_worker_meta()
      → clear_connector_metadata()
  → ModelRunnerOutput(kv_connector_output=...)
  → Scheduler.update_from_output()
```

如果把它压缩成一条链：

```text
kv_connector_metadata
  → Worker connector bind
  → forward context 中 start load
  → attention layer 中 wait load / save layer
  → forward 结束收集 output
  → kv_connector_output
```

---

## 3. Worker 侧 connector 是什么

`KVConnectorBase_V1` 在注释里把 Worker-side methods 单独列出来。

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:27`

Worker 侧接口包括：

```text
handle_preemptions()
start_load_kv()
wait_for_layer_load()
save_kv_layer()
wait_for_save()
get_finished()
build_connector_worker_meta()
get_block_ids_with_load_errors()
register_kv_caches()
register_cross_layers_kv_cache()
set_host_xfer_buffer_ops()
clear_connector_metadata()
shutdown()
```

这些接口围绕一个核心事实展开：

```text
Worker connector 必须能访问本地 KV cache tensor，并且必须嵌入模型 forward / attention layer 的执行时序。
```

这也是为什么 Worker connector 不能只在 Executor 层做普通 RPC：

```text
1. load 的目标是每层 paged KV cache tensor；
2. save 的源也是每层 paged KV cache tensor；
3. async load 需要在 attention layer 使用该层 KV 前同步；
4. async save 需要在 KV block 被覆盖或释放前完成；
5. connector 需要 forward_context 里的 attention metadata、slot mapping、layer 信息。
```

---

## 4. Worker connector 的创建时机

Worker 侧 connector 不是 Scheduler 创建的那个实例，而是在 Worker 进程内单独创建。

入口在 `GPUWorker.initialize_from_config()`：

```python
ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:563`

源码注释解释为什么要在这里初始化：

```text
Init kv cache connector here, because it requires kv_cache_config.
This needs to be done before initialize_kv_cache.
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:570`

`ensure_kv_transfer_initialized()` 的关键逻辑是：

```python
if vllm_config.kv_transfer_config is None:
    return

if vllm_config.kv_transfer_config.is_kv_transfer_instance and _KV_CONNECTOR_AGENT is None:
    _sync_engine_id_across_tp(vllm_config)
    _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(
        config=vllm_config,
        role=KVConnectorRole.WORKER,
        kv_cache_config=kv_cache_config,
    )
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_transfer_state.py:72`

几个关键点：

```text
1. 没有 kv_transfer_config 时不创建；
2. 只有 is_kv_transfer_instance 为 True 的 worker 才创建；
3. 创建的是 role=KVConnectorRole.WORKER；
4. 创建前会在 TP / PP 组内同步 engine_id；
5. connector 存在全局 _KV_CONNECTOR_AGENT 中，通过 get_kv_transfer_group() 获取。
```

所以 Worker 侧 connector 的定位是：

```text
每个需要参与 KV transfer 的 Worker 进程内，都有一个本地 connector agent。
```

---

## 5. Worker connector 如何拿到本地 KV cache tensor

创建 connector 只拿到了配置和 `kv_cache_config`，还没有拿到真正的 GPU KV cache tensor。

真正注册发生在 `GPUModelRunner.initialize_kv_cache()` 末尾。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7303`

主流程是：

```text
initialize_kv_cache(kv_cache_config)
  → initialize_attn_backend(...)
  → initialize_metadata_builders(...)
  → initialize_kv_cache_tensors(...)
  → bind_kv_cache(...)
  → if has_kv_transfer_group() and not is_profiling:
        register_cross_layers_kv_cache(...) 或 register_kv_caches(...)
        set_host_xfer_buffer_ops(copy_kv_blocks)
```

对应代码：

```python
if has_kv_transfer_group() and not is_profiling:
    kv_transfer_group = get_kv_transfer_group()
    if self.cross_layers_kv_cache is not None:
        kv_transfer_group.register_cross_layers_kv_cache(
            self.cross_layers_kv_cache, self.cross_layers_attn_backend
        )
    else:
        kv_transfer_group.register_kv_caches(kv_caches)
    kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7351`

这里说明两种 KV cache 布局：

```text
普通布局：
  connector.register_kv_caches(kv_caches)
  kv_caches 是 layer_name -> kv_cache_tensor 的映射。

cross-layer uniform 布局：
  connector.register_cross_layers_kv_cache(cross_layers_kv_cache, attn_backend)
  多层 KV 放进一个跨层连续 tensor，适合按 block 批量传输。
```

是否使用 cross-layer layout 由 `use_uniform_kv_cache()` 判断。

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:115`

它要求：

```text
1. 当前存在 KV transfer group；
2. connector.prefer_cross_layer_blocks 为 True；
3. 只有一个 attention group / KV cache group；
4. attention backend 支持带 num_layers 维度的 stride order。
```

因此 Worker connector 执行 load / save 的前置条件是：

```text
connector 已经知道本地 KV cache tensor 的地址、形状和 layer 映射。
```

注意新版 `v1/worker/gpu/kv_connector.py:47` 的 `ActiveKVConnector` 当前只注册 `register_kv_caches(kv_caches_dict)`，源码里仍有 TODO 标注 cross-layer KV cache 支持待补；legacy model runner 路径才包含上面这段 `register_cross_layers_kv_cache(...)` 分支。

---

## 6. KVConnectorMetadata 在 Worker 侧如何进入 execute_model

`SchedulerOutput` 定义了：

```python
kv_connector_metadata: KVConnectorMetadata | None = None
```

位置：`code/vllm/vllm/v1/core/sched/output.py:232`

Executor / Worker 只是把这个 `SchedulerOutput` 透传给 `GPUModelRunner.execute_model()`。

在 legacy `GPUModelRunner.execute_model()` 开头，如果存在 KV transfer group，会先取出 metadata：

```python
if has_kv_transfer_group():
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4078`

这一步还没有真正 load KV，而是先处理 preemption / evicted blocks。新版 GPU runner 则把同样的动作放在 `ActiveKVConnector.pre_forward()` 内部，位置：`code/vllm/vllm/v1/worker/gpu/kv_connector.py:61` 到 `code/vllm/vllm/v1/worker/gpu/kv_connector.py:75`。

为什么要在 forward 前先 `handle_preemptions()`？

```text
某些 connector 会异步保存被抢占或即将被覆盖的 blocks。
如果这些 blocks 在保存完成前被新请求复用，就可能把还没保存完的数据覆盖掉。
所以 Worker connector 要在本轮 forward / load 之前先处理这些 preemption metadata。
```

抽象接口说明：

```python
def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
    """
    Handle preempted requests or evicted blocks BEFORE they are overwritten.
    Needed for connectors which use async saves.
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:285`

---

## 7. execute_model 中 KV connector 的主插入点

`GPUModelRunner.execute_model()` 的主线仍然是执行层常规链路：

```text
_update_states()
_prepare_inputs()
_build_attention_metadata()
_preprocess()
_model_forward()
postprocess / logits / pooling / sample_tokens
```

KV connector 的关键插入点在模型 forward 的 context manager 中：

```python
with (
    set_forward_context(attn_metadata, self.vllm_config, ...),
    record_function_or_nullcontext("gpu_model_runner: forward"),
    self.maybe_get_kv_connector_output(
        scheduler_output,
        defer_finalize=defer_kv_connector_finalize,
    ) as kv_connector_output,
):
    model_output = self._model_forward(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4302`

这说明：

```text
KV connector 的 load / save 生命周期包住整个 model forward。
```

顺序很关键：

```text
1. 先 set_forward_context(attn_metadata, ...)
2. 再 maybe_get_kv_connector_output(...)
3. 再执行 _model_forward()
```

原因是：

```text
start_load_kv() 需要 get_forward_context()；
attention layer hook 也需要 forward_context 中的 attention metadata 和 kv_cache。
```

---

## 8. maybe_get_kv_connector_output 做什么

`maybe_get_kv_connector_output()` 定义在 `KVConnectorModelRunnerMixin`。

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:51`

如果没有 KV transfer group，它返回空 context：

```python
return (
    _get_kv_connector_output(...)
    if has_kv_transfer_group()
    else nullcontext()
)
```

真正逻辑在 `_get_kv_connector_output()`。

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:77`

进入 context 时：

```python
output = KVConnectorOutput()
kv_connector = get_kv_transfer_group()
assert scheduler_output.kv_connector_metadata is not None
kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
kv_connector.start_load_kv(get_forward_context())
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:83`

退出 context 时：

```python
if wait_for_save and not defer_finalize:
    kv_connector.wait_for_save()

output.finished_sending, output.finished_recving = (
    kv_connector.get_finished(scheduler_output.finished_req_ids)
)
output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
output.kv_connector_stats = kv_connector.get_kv_connector_stats()
output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()
output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()

if not defer_finalize:
    kv_connector.clear_connector_metadata()
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:99`

因此这个 context manager 同时承担 3 件事：

```text
1. forward 前：绑定 metadata，并启动 load；
2. forward 中：让 attention layer hook 能找到 metadata；
3. forward 后：等待 save、收集 output、清理 metadata。
```

新版 GPU model runner 不再用这个 context manager 包住 forward，而是通过 `v1/worker/gpu/kv_connector.py` 的 `ActiveKVConnector` 拆成显式的 `pre_forward()` / `post_forward()`：`pre_forward()` 调 `handle_preemptions()`、`bind_connector_metadata()` 和 `start_load_kv()`；`post_forward()` 调 `wait_for_save()`、`get_finished()`、`get_block_ids_with_load_errors()`、收集 stats/events/worker_meta 并 `clear_connector_metadata()`。no-forward 路径则直接调用 `ActiveKVConnector.no_forward()`。

---

## 9. bind_connector_metadata 的意义

`bind_connector_metadata()` 是 Worker 侧 connector 消费 Scheduler metadata 的入口。

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:211`

接口注释说明：

```text
This function should be called by the model runner every time before the model execution.
The metadata will be used for runtime KV cache loading and saving.
```

它只是把 metadata 存到 connector 内部：

```python
self._connector_metadata = connector_metadata
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:221`

真正的使用发生在 connector 实现内部，例如：

```text
start_load_kv() 读取 metadata 中的 load requests；
save_kv_layer() 读取 metadata 中的 store requests；
get_finished() 根据 metadata / 内部 job 状态判断哪些 req 完成；
build_connector_worker_meta() 把本轮 worker 内部状态打包回 Scheduler connector。
```

所以 metadata 的生命周期是：

```text
bind_connector_metadata()
  → start_load_kv() / wait_for_layer_load() / save_kv_layer() 使用
  → get_finished() / build_connector_worker_meta() 生成 output
  → clear_connector_metadata()
```

---

## 10. start_load_kv：forward 前启动 KV load

抽象接口：

```python
def start_load_kv(self, forward_context: ForwardContext, **kwargs) -> None:
    """
    Start loading the KV cache from the connector to vLLM's paged KV buffer.
    This is called from the forward context before the forward pass to enable async loading during model execution.
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:292`

这一步的输入是 `ForwardContext`，它包含：

```text
attention metadata；
slot mapping；
batch descriptor；
当前 step 的 token / request 布局；
静态 forward context 中的 attention layers；
```

Worker connector 可以选择：

```text
同步 load：start_load_kv() 直接把外部 KV 拷到本地 paged KV cache；
异步 load：start_load_kv() 发起后台任务，后续在 wait_for_layer_load(layer_name) 再按层同步；
无 load：如果 metadata 没有 load request，则直接返回。
```

注意：

```text
start_load_kv() 是按 step 调一次；wait_for_layer_load() 是按 attention layer 调多次。
```

---

## 11. attention layer hook：为什么 load / save 在层内触发

attention 层通过 `maybe_transfer_kv_layer` 装饰器接入 KV transfer。

位置：`code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:15`

装饰器逻辑是：

```python
if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
    return func(*args, **kwargs)

attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
connector = get_kv_transfer_group()
if attn_metadata is None or not connector.has_connector_metadata():
    return func(*args, **kwargs)

connector.wait_for_layer_load(layer_name)
result = func(*args, **kwargs)
connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
return result
```

位置：`code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:39`

这说明 attention layer 的执行被包成：

```text
进入 attention layer 前：
  wait_for_layer_load(layer_name)

执行 attention：
  读取已有 KV，写入本轮新 KV

退出 attention layer 后：
  save_kv_layer(layer_name, kv_cache, attn_metadata)
```

为什么不是在 model forward 前一次性等完所有层 load？

```text
1. 异步 connector 可以边 forward 前面的计算，边加载后面的层；
2. layer-by-layer pipelining 可以降低等待时间；
3. 某些 connector 只需要保存或加载部分层；
4. attention metadata 和 layer_name 在 attention 层最明确。
```

---

## 12. wait_for_layer_load：保证该层 KV 已经可用

抽象接口：

```python
def wait_for_layer_load(self, layer_name: str) -> None:
    """
    Block until the KV for a specific layer is loaded into vLLM's paged buffer.
    This is called from within attention layer to ensure async copying from start_load_kv is complete.
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:310`

语义是：

```text
在 attention 读取该层历史 KV 前，确保 external KV 已经写入本地 paged KV cache 对应 block / slot。
```

这一步对于 external KV load 非常关键：

```text
如果等待过早，会损失异步 overlap；
如果等待过晚，attention 会读到旧数据或未初始化数据；
所以最精确的位置就是 attention layer entry。
```

---

## 13. save_kv_layer：attention 后保存本层 KV

抽象接口：

```python
def save_kv_layer(
    self,
    layer_name: str,
    kv_layer: torch.Tensor,
    attn_metadata: AttentionMetadata,
    **kwargs,
) -> None:
    """
    Start saving a layer of KV cache from vLLM's paged buffer to the connector.
    This is called from within attention layer to enable async copying during execution.
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:324`

输入里最重要的是：

```text
layer_name：当前 attention layer；
kv_layer：该层本地 paged KV cache tensor；
attn_metadata：包含 slot mapping / block table / query layout 等信息。
```

这一步可以用于：

```text
1. 请求结束后把完整 prompt / generated KV 保存到外部 KVPool；
2. preemption / eviction 前把即将被覆盖的 blocks offload；
3. P/D disaggregation 中 producer 把 KV 推给 consumer；
4. debugging connector 把 KV dump 到磁盘。
```

注意：

```text
save_kv_layer() 通常只是“启动保存”，不一定同步完成。
真正等待所有 save 完成的是 wait_for_save()。
```

---

## 14. wait_for_save：forward 结束时的保存栅栏

抽象接口：

```python
def wait_for_save(self):
    """
    Block until all the save operations is done.
    This prevents overwrites of paged KV buffer before saving done.
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:346`

默认情况下，`_get_kv_connector_output()` 在 forward context 退出时会调用：

```python
kv_connector.wait_for_save()
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:99`

它的作用是：

```text
确保本轮由 save_kv_layer() 发起的必要保存已经完成，避免后续 batch 复用 / 覆盖 paged KV blocks 时破坏外部保存数据。
```

但 speculative decoding 有一个特殊分支：

```python
defer_kv_connector_finalize = self.speculative_config is not None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4301`

如果开启 spec decode，connector finalization 会延后到 `sample_tokens()`：

```python
if spec_config is not None:
    self.finalize_kv_connector()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4599`

原因是：

```text
target model forward 后，draft model 可能还需要保存自己的 KV cache。
所以 wait_for_save + clear metadata 不能太早执行。
```

---

## 15. 没有 scheduled tokens 时为什么仍然要跑 connector

`GPUModelRunner.execute_model()` 中，如果本轮没有 scheduled tokens：

```python
if not num_scheduled_tokens:
    if not has_kv_transfer_group():
        return EMPTY_MODEL_RUNNER_OUTPUT
    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4096`

`kv_connector_no_forward()` 的逻辑是：

```python
with (
    set_forward_context(None, vllm_config),
    _get_kv_connector_output(scheduler_output, wait_for_save=False) as kv_connector_output,
):
    pass

return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:35`

这说明：

```text
即使本轮没有模型 forward，也可能需要推进 KV transfer 状态。
```

典型场景：

```text
1. 异步 KV load 正在后台进行，本轮只是轮询 finished_recving；
2. 异步 KV save 正在后台进行，本轮只是轮询 finished_sending；
3. connector 有 stats / events / worker_meta 需要回传；
4. Scheduler 需要一个 ModelRunnerOutput 来推进 WAITING_FOR_REMOTE_KVS 或 deferred free。
```

为什么 `wait_for_save=False`？

```text
no-forward step 主要用于推进后台状态，不应该无条件阻塞等待所有 save 完成；否则可能把异步 save 退化成同步等待。
```

---

## 16. KVConnectorOutput 是什么

`KVConnectorOutput` 定义在 `v1/outputs.py`。

位置：`code/vllm/vllm/v1/outputs.py:196`

字段包括：

```text
finished_sending
finished_recving
kv_connector_stats
kv_cache_events
kv_connector_worker_meta
invalid_block_ids
expected_finished_count
```

含义分别是：

```text
finished_sending：
  Worker 侧异步 save / send 已完成的 request ids。

finished_recving：
  Worker 侧异步 load / recv 已完成的 request ids。

kv_connector_stats：
  connector 自定义传输统计。

kv_cache_events：
  KV cache event，例如 block stored 等事件。

kv_connector_worker_meta：
  Worker connector 返回给 Scheduler connector 的自定义 metadata。

invalid_block_ids：
  外部 KV load 失败或不可用的本地 block ids，Scheduler 后续据此 recompute 或 fail。

expected_finished_count：
  对 finished_sending / finished_recving 聚合计数的动态配置。
```

`ModelRunnerOutput` 中直接携带：

```python
kv_connector_output: KVConnectorOutput | None = None
```

位置：`code/vllm/vllm/v1/outputs.py:262`

所以 Worker connector 的所有状态回传都收敛到：

```text
ModelRunnerOutput.kv_connector_output
```

---

## 17. get_finished：finished_recving / finished_sending 从哪里来

抽象接口：

```python
def get_finished(
    self, finished_req_ids: set[str]
) -> tuple[set[str] | None, set[str] | None]:
    """
    Returns ids of requests that have finished asynchronous transfer,
    tuple of (sending/saving ids, recving/loading ids).
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:357`

它的输入是：

```text
scheduler_output.finished_req_ids
```

也就是 Scheduler 告诉 Worker：哪些请求在上一轮到这一轮之间已经结束。

Worker connector 可以据此：

```text
1. 检查这些 finished requests 的 async save 是否完成；
2. 检查此前 WAITING_FOR_REMOTE_KVS 的 async load 是否完成；
3. 返回可以让 Scheduler 推进状态的 request ids。
```

`_get_kv_connector_output()` 会把结果写入 output：

```python
output.finished_sending, output.finished_recving = (
    kv_connector.get_finished(scheduler_output.finished_req_ids)
)
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:102`

---

## 18. invalid_block_ids 如何产生

Worker connector 在 load 外部 KV 时可能发现：

```text
1. 外部 KV 被驱逐；
2. 外部读取失败；
3. 校验失败；
4. 部分 block load 失败；
5. 后端返回 key 不存在或数据不完整。
```

抽象接口：

```python
def get_block_ids_with_load_errors(self) -> set[int]:
    """
    Get the set of block IDs that failed to load.
    Applies to both sync- and async-loading requests.
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:375`

接口注释强调：

```text
sync loading：失败应在发现失败的 forward pass 上报；
async loading：失败可在任意 forward pass 上报，但最晚不能晚于该 request 通过 get_finished() 返回的同一 pass。
```

这很重要：

```text
Scheduler 只有收到 invalid_block_ids，才能知道哪些 external-computed blocks 不可信，从而触发 recompute 或 fail。
```

---

## 19. kv_connector_worker_meta 的作用

除了通用字段，connector 还可以返回自定义 worker metadata。

抽象定义：

```python
class KVConnectorWorkerMetadata(ABC):
    """
    Abstract Metadata used to communicate back
    Worker KVConnector -> Scheduler KVConnector.
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:150`

它要求实现：

```python
def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:161`

Worker 侧通过：

```python
output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:109`

Scheduler 侧会先调用：

```python
connector.update_connector_output(kv_connector_output)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2428`

所以 `kv_connector_worker_meta` 的作用是：

```text
给具体 connector 一个扩展通道，用于传回通用 finished_sending / finished_recving / invalid_block_ids 无法表达的状态。
```

例如 offloading connector 可以把 completed jobs、flush fences、transfer stats 等信息放进去。

---

## 20. 多 worker 输出如何聚合

多进程 / 多 worker 后端会收到多个 `ModelRunnerOutput`。

KV connector output 的聚合工具是 `KVOutputAggregator`。

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/utils.py:50`

它的职责是：

```text
把所有 worker 的 kv_connector_output 聚合成一个给 Scheduler 的 output。
```

初始化时会决定每个 request 需要多少个 worker 都报告完成：

```python
@classmethod
def from_connector(cls, connector, world_size):
    return cls(connector.get_finished_count() or world_size)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/utils.py:61`

聚合 finished set 的逻辑是：

```text
每收到一个 worker 报告某 req_id finished，就把 remaining_count 减 1；
只有 remaining_count 变成 0，才把该 req_id 放入最终 finished_sending / finished_recving。
```

对应代码位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/utils.py:75`

聚合内容包括：

```text
finished_sending / finished_recving：按 expected_finished_count 计数聚合；
kv_connector_stats：调用 stats.aggregate()；
kv_connector_worker_meta：调用 worker_meta.aggregate()；
kv_cache_events：合并 events；
invalid_block_ids：取 union；
expected_finished_count：允许 worker 动态更新。
```

最终输出仍然是一个 `KVConnectorOutput`。

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/utils.py:160`

关键点：

```text
Scheduler 看到的是聚合后的 connector output，而不是每个 worker 的原始 output。
```

---

## 21. ExampleConnector 的最小 Worker 侧例子

`ExampleConnector` 是最容易理解的最小实现：它把 KV cache 保存到磁盘，再从磁盘读回。

文件：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py`

### 21.1 metadata 结构

`ExampleConnectorMetadata` 里保存 requests：

```python
@dataclass
class ReqMeta:
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    mm_hashes: list[str]
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:31`

`slot_mapping` 由 `block_ids` 和 `block_size` 展开：

```text
block id + block offset → paged KV cache 中的物理 slot。
```

这说明 Worker connector 最终需要知道的不是抽象 token 命中，而是：

```text
要把哪些 token 对应的 KV 放进哪些物理 slot。
```

### 21.2 start_load_kv：从磁盘注入 KV

`start_load_kv()` 中先取 metadata：

```python
metadata = self._get_connector_metadata()
assert isinstance(metadata, ExampleConnectorMetadata)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:151`

然后遍历 metadata 中 `is_store=False` 的请求：

```python
for request in metadata.requests:
    if request.is_store:
        continue
    for layer_name in forward_context.no_compile_layers:
        layer = forward_context.no_compile_layers[layer_name]
        kv_cache_layer = getattr(layer, "kv_cache", None)
        ...
        kv_cache = safetensors.torch.load_file(filename, device=str(kv_cache_layer.device))["kv_cache"]
        inject_kv_into_layer(kv_cache_layer, kv_cache, request.slot_mapping, attn_metadata[layer_name])
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:160`

它展示了 load 的本质：

```text
外部 KV 数据 → 根据 slot_mapping 写入本地 paged KV cache tensor。
```

### 21.3 save_kv_layer：从本地 KV cache 抽取并保存

`save_kv_layer()` 中遍历 `is_store=True` 的请求：

```python
for request in connector_metadata.requests:
    if request.is_store:
        filename = self._generate_filename_debug(...)
        kv_cache = extract_kv_from_layer(kv_layer, request.slot_mapping)
        tensors = {"kv_cache": kv_cache.detach().cpu()}
        safetensors.torch.save_file(tensors, filename)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:237`

它展示了 save 的本质：

```text
本地 paged KV cache tensor → 根据 slot_mapping 抽取请求 KV → 写到外部存储。
```

### 21.4 ExampleConnector 的局限

ExampleConnector 是 debug 实现：

```text
1. 使用磁盘 safetensors；
2. load / save 基本同步；
3. 没有复杂异步线程；
4. 没有跨 worker handshake；
5. 主要用于说明接口语义。
```

真实 KVPool / Mooncake / NIXL / LMCache connector 会把同样的接口映射到 RDMA、共享内存、磁盘 staging、后台线程、远端 KV server 等机制。

---

## 22. MooncakeStore / KVPool 类 Worker connector 的定位

`MooncakeStoreConnector` 更接近 KVPool 场景。

Worker 侧文件：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`

它包含：

```text
store worker；
transfer threads；
lookup server；
MooncakeDistributedStore integration；
ExternalCachedBlockPool；
ChunkedTokenDatabase；
ReqMeta / PoolKey / KeyMetadata；
RDMA / disk staging 相关配置。
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:7`

和 ExampleConnector 相比，MooncakeStore Worker 侧更复杂，是因为它要处理：

```text
1. 外部 KVPool 的 key / metadata / block chunk 映射；
2. RDMA 或 standalone-store / embedded-store 的传输拓扑；
3. 发送线程和接收线程；
4. 外部缓存 block pool；
5. 磁盘 offload staging buffer；
6. 多 rank / TP / DP / PP 下的地址和 engine id 协调；
7. 异步 load / save 完成状态；
8. load 失败或 external block 不可用时的 invalid block 上报。
```

但无论内部多复杂，对 ModelRunner 暴露的仍然是同一组接口：

```text
bind_connector_metadata()
start_load_kv()
wait_for_layer_load()
save_kv_layer()
wait_for_save()
get_finished()
get_block_ids_with_load_errors()
build_connector_worker_meta()
clear_connector_metadata()
```

这就是 vLLM KV Connector 抽象的核心价值：

```text
ModelRunner 不需要知道外部 KVPool 的具体协议，只需要在正确的 forward / attention 时序调用统一接口。
```

---

## 23. load 路径完整时序

以外部 KV 命中并需要 Worker load 为例，完整时序是：

```text
Scheduler.schedule()
  → connector.get_num_new_matched_tokens()
  → KVCacheManager.allocate_slots(... num_external_computed_tokens ...)
  → connector.update_state_after_alloc(request, blocks, external_tokens)
  → connector.build_connector_meta(scheduler_output)
  → scheduler_output.kv_connector_metadata

Worker / ModelRunner
  → execute_model(scheduler_output)
  → handle_preemptions(kv_connector_metadata)
  → _update_states()
  → _prepare_inputs()
  → _build_attention_metadata()
  → set_forward_context(attn_metadata, ...)
  → bind_connector_metadata(kv_connector_metadata)
  → start_load_kv(forward_context)
  → model forward
      → each attention layer:
          wait_for_layer_load(layer_name)
          attention forward reads loaded KV
          save_kv_layer(layer_name, kv_cache, attn_metadata)
  → get_finished()
  → get_block_ids_with_load_errors()
  → kv_connector_output

Scheduler.update_from_output()
  → connector.update_connector_output(kv_connector_output)
  → finished_recving → 恢复 WAITING_FOR_REMOTE_KVS
  → invalid_block_ids → recompute 或 fail
```

这里有两个 load 分支：

```text
同步 load：
  start_load_kv() / wait_for_layer_load() 在同一 execute_model 中完成，forward 可以继续跑本轮剩余 token。

异步 load：
  Scheduler 本轮可能 num_new_tokens=0，请求进入 WAITING_FOR_REMOTE_KVS；Worker 侧推进后台 load，之后通过 finished_recving 告诉 Scheduler 可以恢复。
```

---

## 24. save 路径完整时序

以请求结束后需要保存 KV 到外部 KVPool 为例，完整时序是：

```text
Request finished
  → Scheduler._free_request()
  → Scheduler._connector_finished(request)
  → connector.request_finished(...) / request_finished_all_groups(...)
  → 如果需要保存：connector 记录 save 计划，可能要求 delay free blocks
  → 后续 SchedulerOutput.kv_connector_metadata 携带 store metadata

Worker / ModelRunner
  → execute_model(scheduler_output)
  → bind_connector_metadata(store metadata)
  → start_load_kv(forward_context) 可能 no-op
  → 部分 connector 在 attention layer forward 后通过 save_kv_layer(layer_name, kv_cache, attn_metadata) 保存
  → NIXL push 等 connector 也可能通过 metadata / no-forward step / writer thread 推进保存
  → wait_for_save()
  → get_finished(finished_req_ids)
  → kv_connector_output.finished_sending

Scheduler.update_from_output()
  → finished_sending
  → _free_blocks(request)
```

关键点：

```text
Scheduler 决定“这个请求结束后是否要 save，以及 blocks 是否延迟释放”；
Worker 决定“每层 KV 怎么从 tensor 中取出来并写到外部系统”；
Scheduler 等 Worker 报告 finished_sending 后，才真正释放被延迟保护的 blocks。
```

---

## 25. preemption / eviction 路径为什么在 forward 前处理

某些 connector 使用异步 save 或 offload。

如果一个 request 被 preempt，或者某些 blocks 即将被驱逐 / 覆盖，则 connector 可能需要先把这些 blocks 的 KV 保存出去。

Worker 侧入口是：

```python
get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4078`

抽象接口注释明确说：

```text
Handle preempted requests or evicted blocks BEFORE they are overwritten.
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:285`

所以它必须发生在：

```text
本轮 _prepare_inputs / forward / 新 KV 写入之前。
```

否则可能出现：

```text
1. Scheduler 认为旧 block 还能 offload；
2. Worker 已经把旧 block 复用于新请求；
3. connector save 出去的是新请求的数据；
4. 后续 external KV 命中读回错误内容。
```

这也是 deferred free / async safety 需要单独专题的原因。

---

## 26. Pipeline Parallel / sample_tokens 下的 KV connector output

`execute_model()` 不一定直接返回最终 `ModelRunnerOutput`。

常见生成模型路径中：

```text
execute_model()
  → 完成 forward / logits
  → self.execute_model_state = ExecuteModelState(...)
  → self.kv_connector_output = kv_connector_output
  → return None

sample_tokens()
  → 消费 execute_model_state
  → 采样 / bookkeeping
  → ModelRunnerOutput(kv_connector_output=self.kv_connector_output)
```

相关位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4386`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4405`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4605`

PP 非最后一 rank 也有特殊处理：

```python
if not get_pp_group().is_last_rank:
    self.kv_connector_output = kv_connector_output
    return hidden_states
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4339`

如果 `sample_tokens()` 被调用时没有 `execute_model_state`，会返回 only connector output：

```python
return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4426`

这说明：

```text
KV connector output 的生命周期不完全等同于 sampled token output；
它可能随 execute_model 保存，最后由 sample_tokens 或 PP pass-through 带回 Scheduler。
```

---

## 27. Worker connector 和 attention metadata / slot mapping 的关系

Worker connector 的核心问题不是“请求命中了多少 token”，而是：

```text
这些 token 的 KV 在本地 paged KV cache 中对应哪些物理位置。
```

这个物理位置由 Scheduler 分配 blocks，ModelRunner 构造 slot mapping。

在 ExampleConnector 中：

```python
slot_mapping = block_offsets + block_ids * block_size
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:54`

在 attention hook 中：

```python
attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
```

位置：`code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:45`

所以 Worker connector 同时依赖两类信息：

```text
Scheduler metadata：
  哪些 request 需要 load / save，相关 block ids / tokens / keys 是什么。

Forward context / attention metadata：
  当前 forward 中每层 KV cache tensor、slot mapping、attention backend 布局是什么。
```

缺少前者，不知道要搬哪些 KV；缺少后者，不知道怎么把 KV 正确写入 / 读出本地 tensor。

---

## 28. Worker connector 和 KV cache layout 的关系

普通 KV cache layout 下，connector 看到的是：

```text
layer_name -> kv_cache tensor
```

注册入口：

```python
register_kv_caches(kv_caches)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:251`

cross-layer layout 下，connector 看到的是：

```text
一个带 num_layers 维度的 cross_layers_kv_cache tensor
```

注册入口：

```python
register_cross_layers_kv_cache(kv_cache, attn_backend)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:261`

为什么需要 cross-layer layout？

```text
如果一个 block 的所有层 KV 在内存上更连续，connector 可以按 block 批量搬运多层 KV，减少 per-layer 小拷贝和调度开销。
```

但它只有在 connector 和 attention backend 都支持时才启用。

---

## 29. Worker 侧 shutdown

Worker 关闭时会清理 KV connector。

`GPUWorker.shutdown()` 中：

```python
if ensure_kv_transfer_shutdown is not None:
    ensure_kv_transfer_shutdown()
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:1141`

`ensure_kv_transfer_shutdown()` 做：

```python
if _KV_CONNECTOR_AGENT is not None:
    _KV_CONNECTOR_AGENT.shutdown()
    _KV_CONNECTOR_AGENT = None
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_transfer_state.py:97`

作用是：

```text
1. 等待或终止 connector 内部后台传输；
2. 释放外部连接 / RDMA / staging buffer / thread；
3. 清空全局 connector agent；
4. 避免 worker 进程退出时留下未完成资源。
```

---

## 30. 一个完整例子：同步 external KV load + 本轮 forward

假设：

```text
prompt = 10000 tokens
local prefix hit = 3000 tokens
external KV hit beyond local = 5000 tokens
load_kv_async = False
本轮需要本地 forward 剩余 2000 tokens
```

Scheduler 侧已经完成：

```text
1. 分配 local prefix blocks；
2. 为 external 5000 tokens 分配本地 blocks；
3. 为 new 2000 tokens 分配本地 blocks；
4. 构造 kv_connector_metadata。
```

Worker 侧执行：

```text
1. execute_model() 收到 SchedulerOutput；
2. handle_preemptions(metadata)；
3. _update_states() 把请求加入 input_batch；
4. _prepare_inputs() 生成 positions / slot mapping；
5. _build_attention_metadata() 生成 attention metadata；
6. bind_connector_metadata(metadata)；
7. start_load_kv(forward_context) 发起 external KV load；
8. attention layer entry 调 wait_for_layer_load(layer_name)，确保该层前 8000 tokens 的 KV 已在本地 slots；
9. attention forward 只计算剩余 2000 tokens，并把新 KV 写入 blocks；
10. attention layer exit 调 save_kv_layer()，如 metadata 要求 store 则保存；
11. forward 结束 wait_for_save()；
12. 返回 kv_connector_output。
```

关键点：

```text
Worker forward 不重新计算 external hit 的 5000 tokens；
但 attention 必须能在本地 paged KV cache 中读到这些 token 的 KV。
```

---

## 31. 一个完整例子：异步 external KV load，无本轮 forward

假设：

```text
prompt = 10000 tokens
external KV hit = 8000 tokens
load_kv_async = True
Scheduler 本轮 num_new_tokens = 0
request.status = WAITING_FOR_REMOTE_KVS
```

Worker 侧可能经历：

```text
1. execute_model() 收到 scheduler_output；
2. total_num_scheduled_tokens = 0；
3. 因为 has_kv_transfer_group()，不返回 EMPTY_MODEL_RUNNER_OUTPUT；
4. 进入 kv_connector_no_forward()；
5. bind_connector_metadata(metadata)；
6. start_load_kv(forward_context=None) 发起或推进后台 load；
7. 不执行模型 forward；
8. get_finished() 如果 load 完成，返回 finished_recving；
9. get_block_ids_with_load_errors() 如果 load 失败，返回 invalid_block_ids；
10. ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output) 回 Scheduler。
```

Scheduler 收到后：

```text
finished_recving：
  下一轮把 WAITING_FOR_REMOTE_KVS 恢复为 WAITING / PREEMPTED。

invalid_block_ids：
  根据 kv_load_failure_policy 选择 recompute 或 fail。
```

关键点：

```text
异步 load 的推进不依赖本轮有模型 forward。
```

---

## 32. 一个完整例子：请求结束后异步 save

假设：

```text
request 已完成；
Scheduler connector.request_finished() 决定需要保存 KV 到外部 KVPool；
connector 返回 delay free blocks。
```

Worker 侧后续某轮收到 store metadata：

```text
1. bind_connector_metadata(store metadata)；
2. model forward 或 no-forward step 进入 connector 生命周期；
3. attention layer exit 调 save_kv_layer()；
4. connector 后台保存该层 KV；
5. wait_for_save() 根据 connector 策略等待必要保存完成；
6. get_finished(finished_req_ids) 返回 finished_sending；
7. Scheduler 收到 finished_sending 后真正 free blocks。
```

如果 connector 把保存状态放在 `kv_connector_worker_meta` 而不是 `finished_sending`，Scheduler 侧 connector 会在 `update_connector_output()` 中消费这些自定义状态。

---

## 33. Worker 侧和 Scheduler 侧的边界

Scheduler 侧 connector 负责：

```text
外部命中查询；
load_kv_async 决策；
分配后记录 block ids；
build_connector_meta()；
request_finished() / request_finished_all_groups()；
update_connector_output()；
根据 Worker output 更新 connector 内部状态。
```

Worker 侧 connector 负责：

```text
注册本地 KV cache tensor；
绑定本轮 metadata；
启动 load；
在 attention layer 前等待 load；
在 attention layer 后保存 KV；
等待 save；
报告 finished / invalid / stats / worker_meta；
关闭后台资源。
```

Scheduler 主逻辑负责：

```text
waiting / running / preempted 状态迁移；
KVCacheManager block 分配和释放；
WAITING_FOR_REMOTE_KVS 恢复；
invalid blocks 的 recompute / fail；
deferred free。
```

边界一句话：

```text
Scheduler 决定“哪些 KV 应该搬、搬到哪些 blocks、何时恢复请求”；Worker 负责“把这些 KV 真实搬进 / 搬出本地 tensor”。
```

---

## 34. 容易疑惑的点

### 34.1 Worker connector 会重新查询外部命中吗？

通常不会。

```text
外部命中查询属于 Scheduler 侧 get_num_new_matched_tokens()。
Worker 侧只消费 Scheduler 已经封装好的 metadata。
```

### 34.2 start_load_kv 一定会同步完成 load 吗？

不一定。

```text
它可以同步完成，也可以只发起异步任务。
如果异步，attention layer 会在 wait_for_layer_load(layer_name) 精确等待当前层。
```

### 34.3 save_kv_layer 一定表示请求已经结束吗？

不一定。

```text
它只是 attention layer 退出时的 hook。
具体是否保存当前层 KV，取决于 connector metadata 中是否有 store / offload / preemption 计划。
```

### 34.4 为什么 no-forward 还要返回 ModelRunnerOutput？

因为：

```text
异步 KV transfer 的完成状态也需要通过 ModelRunnerOutput.kv_connector_output 回 Scheduler。
即使没有 sampled tokens，也可能有 finished_recving / finished_sending / invalid_block_ids。
```

### 34.5 invalid_block_ids 是 request id 吗？

不是。

```text
invalid_block_ids 是本地 block ids。
Scheduler 会根据这些 block ids 找到受影响请求，再按策略 recompute 或 fail。
```

### 34.6 kv_connector_worker_meta 和 finished_recving / finished_sending 重复吗？

不完全重复。

```text
finished_recving / finished_sending 是通用完成信号；
kv_connector_worker_meta 是 connector 自定义扩展通道，用于传更复杂的状态。
```

### 34.7 Worker connector metadata 什么时候清理？

通常在 `_get_kv_connector_output()` 退出时：

```text
wait_for_save → collect output → clear_connector_metadata
```

如果 spec decode 设置了 `defer_finalize=True`，则延后到 `sample_tokens()` 中的 `finalize_kv_connector()`。

---

## 35. 从“回答问题”的角度总结

如果问：

```text
Worker / ModelRunner 如何消费 KV Connector metadata？
```

可以回答：

```text
Worker 侧在 initialize_from_config() 中根据 kv_transfer_config 创建 role=WORKER 的 connector，并在 ModelRunner 初始化 KV cache 后把本地 paged KV cache tensors 注册给 connector。每轮 execute_model() 收到 SchedulerOutput 后，GPUModelRunner 先用 kv_connector_metadata 处理 preemption，然后在 set_forward_context(attn_metadata, ...) 之后进入 maybe_get_kv_connector_output()。这个 context 会 bind_connector_metadata()，再调用 start_load_kv(forward_context) 启动外部 KV load。模型 forward 期间，每个 attention layer 会通过 maybe_transfer_kv_layer 在进入时调用 wait_for_layer_load(layer_name)，退出时调用 save_kv_layer(layer_name, kv_cache, attn_metadata)。forward 结束后，context 调 wait_for_save()、get_finished()、get_block_ids_with_load_errors()、build_connector_worker_meta()，形成 KVConnectorOutput，并通过 ModelRunnerOutput.kv_connector_output 返回 Scheduler。
```

如果问：

```text
Worker 侧 KV load / save 发生在哪里？
```

可以回答：

```text
load 的启动点在 ModelRunner forward context 进入时的 start_load_kv()，每层真正使用前在 attention layer entry 的 wait_for_layer_load(layer_name) 同步；save 的触发点在 attention layer exit 的 save_kv_layer(layer_name, kv_cache, attn_metadata)，forward context 退出时通过 wait_for_save() 做必要栅栏。
```

如果问：

```text
ModelRunnerOutput.kv_connector_output 包含什么？
```

可以回答：

```text
它包含 Worker connector 本轮向 Scheduler 汇报的状态：finished_recving 表示异步 load 完成的请求，finished_sending 表示异步 save / send 完成的请求，invalid_block_ids 表示 load 失败或不可用的本地 blocks，kv_connector_stats / kv_cache_events 用于统计和事件，kv_connector_worker_meta 用于 connector 自定义状态，expected_finished_count 用于多 worker 完成计数聚合。
```

---

## 36. 最关键流程图

```text
Worker 初始化
  ├─ GPUWorker.initialize_from_config(kv_cache_config)
  │    └─ ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
  │         └─ KVConnectorFactory.create_connector(role=WORKER)
  │
  └─ GPUModelRunner.initialize_kv_cache(kv_cache_config)
       ├─ initialize_kv_cache_tensors(...)
       ├─ bind_kv_cache(...)
       └─ connector.register_kv_caches(...) / register_cross_layers_kv_cache(...)

每轮执行
  ├─ GPUModelRunner.execute_model(scheduler_output)
  │
  ├─ if has_kv_transfer_group:
  │    └─ handle_preemptions(scheduler_output.kv_connector_metadata)
  │
  ├─ _update_states(scheduler_output)
  ├─ _prepare_inputs(...)
  ├─ _build_attention_metadata(...)
  ├─ _preprocess(...)
  │
  └─ with set_forward_context(attn_metadata, ...), maybe_get_kv_connector_output(...):
       ├─ bind_connector_metadata(kv_connector_metadata)
       ├─ start_load_kv(forward_context)
       │
       ├─ model forward
       │    └─ each attention layer
       │         ├─ wait_for_layer_load(layer_name)
       │         ├─ attention forward
       │         └─ save_kv_layer(layer_name, kv_cache, attn_metadata)
       │
       ├─ wait_for_save()
       ├─ get_finished(finished_req_ids)
       ├─ get_block_ids_with_load_errors()
       ├─ get_kv_connector_stats()
       ├─ get_kv_connector_kv_cache_events()
       ├─ build_connector_worker_meta()
       └─ clear_connector_metadata()

输出回收
  ├─ ModelRunnerOutput.kv_connector_output
  ├─ KVOutputAggregator.aggregate()  # 多 worker 时
  └─ Scheduler.update_from_output()
       ├─ connector.update_connector_output(kv_connector_output)
       ├─ finished_recving → 恢复 WAITING_FOR_REMOTE_KVS
       ├─ finished_sending → 释放延迟 blocks
       └─ invalid_block_ids → recompute / fail
```

---

## 37. 最关键对象关系

```text
KVTransferConfig
  控制当前 worker 是否是 KV transfer instance，以及使用哪个 connector。

_KV_CONNECTOR_AGENT
  Worker 进程内全局 connector agent，通过 get_kv_transfer_group() 访问。

KVConnectorRole.WORKER
  Worker 侧 connector 角色，区别于 Scheduler 侧 role=SCHEDULER。

KVConnectorMetadata
  Scheduler connector → Worker connector 的本轮 KV transfer 计划。

ForwardContext
  ModelRunner 在 forward 前设置的上下文，提供 attention metadata、slot mapping、batch descriptor 等执行信息。

AttentionMetadata
  attention backend 所需元数据，也是 save_kv_layer() 抽取 KV 的关键输入。

KV cache tensors
  Worker 本地真实 paged KV cache，connector load / save 的目标和来源。

KVConnectorOutput
  Worker connector → Scheduler 的通用输出容器。

KVConnectorWorkerMetadata
  Worker connector → Scheduler connector 的自定义扩展 metadata。

KVOutputAggregator
  多 worker 场景下聚合每个 worker 的 KVConnectorOutput。

ModelRunnerOutput
  执行层回传 Scheduler 的结果容器，携带 kv_connector_output。
```

---

## 38. 和其他专题的关系

本篇回答的是 Worker / ModelRunner 侧如何消费 KV connector metadata。

相关专题分工：

```text
04_scheduler_kv_connector_flow.md
  讲 Scheduler 侧如何查询外部命中、分配 blocks、构造 metadata。

05_external_kv_load_flow.md
  讲 external KV load、WAITING_FOR_REMOTE_KVS、finished_recving 和恢复调度。

06_external_kv_save_flow.md
  讲 request_finished、save KV、finished_sending 和延迟释放。

08_invalid_blocks_and_recompute.md
  讲 invalid_block_ids、load failure policy、recompute / fail。

09_deferred_free_and_async_safety.md
  讲 async scheduling / PP 下为什么要 deferred free 防止 block 复用竞态。

10_kvpool_end_to_end.md
  从 KVPool / MooncakeStore 角度串联 Scheduler 和 Worker 的端到端链路。
```

最终最小心智模型：

```text
SchedulerOutput.kv_connector_metadata 是“计划”；Worker connector 绑定它后，在 forward context 中启动 load，在 attention layer 边界等待 load 和触发 save，forward 结束后把完成 / 失败 / 统计状态打包成 ModelRunnerOutput.kv_connector_output；Scheduler 再用这个 output 恢复等待请求、释放延迟 blocks 或触发重算。
```
