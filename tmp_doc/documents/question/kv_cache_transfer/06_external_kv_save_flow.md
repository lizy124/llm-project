# 06. 请求结束后 KV 如何保存到外部 KVPool？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\output.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\kv_connector_model_runner_mixin.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\outputs.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\base.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\lmcache_connector.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\nixl\connector.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\nixl\base_scheduler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\nixl\push_scheduler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\nixl\push_worker.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\offloading\scheduler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\offloading\worker.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\attention\kv_transfer_utils.py`

本文梳理请求结束时 external KV save 的完整路径：Scheduler 如何在释放请求前调用 connector，connector 如何决定是否延迟释放 blocks，Worker 侧如何执行 KV 保存 / 发送，以及 `finished_sending` 如何让 Scheduler 最终释放 KV blocks。

---

## 1. 一句话回答

请求结束后，Scheduler 不会直接无条件释放 KV blocks；如果配置了 KV Connector，它会先调用 connector 的 `request_finished()` / `request_finished_all_groups()`，让 connector 有机会把本地 KV blocks 保存或发送到外部 KVPool / 远端节点。

主链路是：

```text
Scheduler.update_from_output()
  → request stop
  → _handle_stopped_request()
  → _free_request()
      → _connector_finished()
          → remove_skipped_blocks()
          → get_block_ids()
          → connector.request_finished(...)
             或 connector.request_finished_all_groups(...)
      → connector_delay_free_blocks ? 暂不释放 : _free_blocks()
  → 下一轮 schedule()
      → connector.build_connector_meta()
      → SchedulerOutput.kv_connector_metadata
  → Worker / ModelRunner execute_model()
      → bind_connector_metadata()
      → start_load_kv() / save_kv_layer() / wait_for_save()
      → get_finished(finished_req_ids)
      → ModelRunnerOutput.kv_connector_output.finished_sending
  → Scheduler.update_from_output()
      → _update_from_kv_xfer_finished()
      → _free_blocks()
```

所以：

```text
request_finished() 是 Scheduler 侧“请求结束、blocks 即将释放前”的保存决策点；
finished_sending 是 Worker 侧“异步保存/发送已完成，可以最终释放 blocks”的回执。
```

---

## 2. 本文要回答的问题

```text
请求什么时候触发 external KV save？
Scheduler._free_request() 和 _connector_finished() 做什么？
request_finished() / request_finished_all_groups() 返回什么？
connector_delay_free_blocks 如何影响 block 释放？
Worker 侧 KV save 在哪里发生？
finished_sending 如何回到 Scheduler？
NIXL push / LMCache / Offloading 的 save 语义有什么差异？
KV save 和 prefix cache / sliding window / deferred free 有什么关系？
```

---

## 3. 最小心智模型

可以把 external KV save 分成两类：

```text
A. 请求结束时保存 / 发送：
   Scheduler 在 _free_request() 里调用 connector.request_finished*()；
   connector 如果要异步保存，就返回 delay_free_blocks=True；
   Scheduler 暂时保留 request 和 blocks，等 finished_sending 后释放。

B. 执行过程中逐步 store：
   Worker 在 attention forward 后调用 connector.save_kv_layer()；
   connector 在 forward context 退出时 wait_for_save() 或把完成信息通过 worker meta 回传；
   Scheduler 根据 connector 类型维护 store job / flush / block reuse 安全。
```

本文标题里的“请求结束后保存”主要指 A，但实际 vLLM connector API 同时支持 A 和 B。

---

## 4. KV Connector 抽象层如何定义 save

抽象定义在：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`

文件顶部注释把接口分成 Scheduler-side 和 Worker-side。

### 4.1 Scheduler 侧请求结束钩子

`KVConnectorBase_V1` 注释说明：

```text
request_finished() - called once when a request is finished,
with the computed kv cache blocks for the request.
Returns whether KV cache should be freed now or if the
connector now assumes responsibility for freeing the blocks asynchronously.
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:17` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:22`

普通单 group 接口：

```python
def request_finished(
    self,
    request: Request,
    block_ids: list[int],
) -> tuple[bool, dict[str, Any] | None]:
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:542`

HMA / 多 KV group 接口：

```python
def request_finished_all_groups(
    self,
    request: Request,
    block_ids: tuple[list[int], ...],
) -> tuple[bool, dict[str, Any] | None]:
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:92` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:97`

返回值含义：

```text
(bool delay_free_blocks, dict | None kv_transfer_params)

bool=True：connector 正在异步 save/send，Scheduler 不能立即释放 blocks；
bool=False：connector 不需要保留 blocks，Scheduler 可以正常释放；
kv_transfer_params：随 EngineCoreOutput 返回给上层，用于远端 decode/prefill 后续请求。
```

### 4.2 Worker 侧保存接口

Worker-side 抽象里有：

```text
save_kv_layer() - starts saving KV for layer i (maybe async)
wait_for_save() - blocks until all saves are done
get_finished() - called with ids of finished requests, returns ids of requests that have completed async sending/recving
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:31` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:39`

接口定义：

```python
def save_kv_layer(
    self,
    layer_name: str,
    kv_layer: torch.Tensor,
    attn_metadata: AttentionMetadata,
    **kwargs: Any,
) -> None:
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:324` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:344`

```python
def wait_for_save(self):
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:346` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:355`

```python
def get_finished(
    self, finished_req_ids: set[str]
) -> tuple[set[str] | None, set[str] | None]:
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:357` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:373`

---

## 5. 请求正常结束时的 Scheduler 主链路

### 5.1 update_from_output 中检测 stopped

模型执行结束后，Scheduler 在 `update_from_output()` 中遍历本轮调度请求。

入口：`code/vllm/vllm/v1/core/sched/scheduler.py:1463`

当请求生成新 token 后，会更新请求状态并检查 stop：

```python
if new_token_ids:
    new_token_ids, stopped = self._update_request_with_output(
        request, new_token_ids
    )
elif request.pooling_params and pooler_output is not None:
    request.status = RequestStatus.FINISHED_STOPPED
    stopped = True
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1588` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1596`

如果 stopped：

```python
if stopped:
    finish_reason = request.get_finished_reason()
    finished = self._handle_stopped_request(request)
    if finished:
        kv_transfer_params = self._free_request(request)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1655` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1663`

含义：

```text
请求 stop 后不会马上 del request；
先经过 _handle_stopped_request()；
只有确认是真的 finished，才进入 _free_request()。
```

### 5.2 resumable / streaming 请求不一定马上 finished

`_handle_stopped_request()`：

```python
def _handle_stopped_request(self, request: Request) -> bool:
    """Return True if finished (can be False for resumable requests)."""
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1830`

如果请求不是 resumable，直接返回 True。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1831` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1833`

如果是 streaming session，还有下一段输入，则会更新 session 并重新入队，不触发最终释放。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1835` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1846`

所以：

```text
只有请求真正结束时，才会触发 _free_request() 和 connector.request_finished*()。
```

---

## 6. abort / 外部 finish 时的 Scheduler 主链路

除了正常生成 stop，API server 断连、abort、错误等会走 `finish_requests()`。

入口：`code/vllm/vllm/v1/core/sched/scheduler.py:1983`

`finish_requests()` 会：

```text
1. 从 running / waiting / skipped_waiting 队列移除请求；
2. 设置 request.status = finished_status；
3. 调用 _free_request(request, delay_free_blocks=...)
```

关键代码：

```python
request.status = finished_status
self._free_request(request, delay_free_blocks=delay_free_blocks)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2041` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2042`

如果请求正在 `WAITING_FOR_REMOTE_KVS`，abort 时可能还在异步 load，本地 blocks 不能立即释放：

```python
if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
    delay_free_blocks = (
        request.request_id not in self.finished_recving_kv_req_ids
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2033` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2037`

这和 save 的延迟释放是同一种安全原则：

```text
只要外部 transfer 还可能读写这些 blocks，就不能把 blocks 还给 BlockPool。
```

---

## 7. _free_request() 做什么

入口：`code/vllm/vllm/v1/core/sched/scheduler.py:2046`

核心代码：

```python
def _free_request(
    self, request: Request, delay_free_blocks: bool = False
) -> dict[str, Any] | None:
    assert request.is_finished()

    self._inflight_prefills.discard(request)
    connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
    self.encoder_cache_manager.free(request)
    request_id = request.request_id
    self.finished_req_ids.add(request_id)
    ...

    delay_free_blocks |= connector_delay_free_blocks
    if not delay_free_blocks:
        self._free_blocks(request)

    return kv_xfer_params
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2046` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2063`

它做四件事：

```text
1. 从 in-flight prefill 集合移除请求；
2. 调用 _connector_finished()，让 KV connector 决定是否保存/发送 KV；
3. 释放 encoder cache；
4. 如果不需要延迟释放 blocks，就调用 _free_blocks()。
```

注意：

```text
finished_req_ids 会加入 SchedulerOutput.finished_req_ids，通知 Worker 清理请求状态；
但如果 connector_delay_free_blocks=True，请求仍保留在 self.requests 里，直到 finished_sending 回来。
```

---

## 8. _connector_finished() 做什么

入口：`code/vllm/vllm/v1/core/sched/scheduler.py:2299`

### 8.1 没有 connector 时直接返回

```python
if self.connector is None:
    return False, None
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2308` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2309`

含义：

```text
没有 KV connector，就没有 external KV save，blocks 走普通释放流程。
```

### 8.2 先移除窗口外 blocks

```python
self.kv_cache_manager.remove_skipped_blocks(
    request_id=request.request_id,
    total_computed_tokens=request.num_computed_tokens,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2311` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2316`

这个步骤很关键：

```text
在把 block table 交给 connector 之前，先让 KVCacheManager 清理 attention 不再需要的 skipped blocks；
例如 sliding window / chunked local attention / Mamba 可能不需要保存整个历史序列的所有 blocks。
```

### 8.3 读取 request 当前 block_ids

```python
block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2318`

此时拿到的是：

```text
request 结束时仍然归它持有、且 connector 应该看到的 KV block table。
```

### 8.4 根据是否 SupportsHMA 选择接口

如果 connector 不支持 HMA：

```python
assert len(self.kv_cache_config.kv_cache_groups) == 1
return self.connector.request_finished(request, block_ids[0])
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2320` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2326`

如果支持 HMA：

```python
return self.connector.request_finished_all_groups(request, block_ids)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2328`

因此：

```text
单 group 旧接口：request_finished(request, list[int])
多 group / HMA 接口：request_finished_all_groups(request, tuple[list[int], ...])
```

---

## 9. connector_delay_free_blocks 如何影响释放

`_connector_finished()` 返回的第一个值会并入 `delay_free_blocks`：

```python
delay_free_blocks |= connector_delay_free_blocks
if not delay_free_blocks:
    self._free_blocks(request)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2059` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2061`

如果为 False：

```text
Scheduler 立即调用 _free_blocks(request)
  → _free_request_blocks(request)
  → del self.requests[request_id]
```

`_free_blocks()`：

```python
def _free_blocks(self, request: Request):
    assert request.is_finished()
    self._free_request_blocks(request)
    del self.requests[request.request_id]
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2065` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2069`

如果为 True：

```text
Scheduler 不调用 _free_blocks()；
request 继续留在 self.requests；
blocks 继续被 request 持有；
等待后续 Worker 汇报 finished_sending；
_update_from_kv_xfer_finished() 再调用 _free_blocks()。
```

这就是 external KV save 的核心安全机制：

```text
异步保存期间，GPU block 不能被 BlockPool 重新分配，否则外部 transfer 可能读到被覆盖的数据。
```

---

## 10. Worker 侧如何执行 KV save / send

### 10.1 SchedulerOutput 携带 connector metadata

每轮 schedule 末尾，如果有 connector：

```python
meta = self._build_kv_connector_meta(self.connector, scheduler_output)
scheduler_output.kv_connector_metadata = meta
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1076` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1082`

`_build_kv_connector_meta()` 本质调用：

```python
return connector.build_connector_meta(scheduler_output)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1100` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1103`

所以请求结束后，如果 connector 在 `request_finished()` 里记录了“要保存/发送哪些 blocks”，这些状态会在下一轮 `build_connector_meta()` 里进入 `SchedulerOutput.kv_connector_metadata`。

### 10.2 GPUModelRunner 绑定 metadata 并执行 connector 生命周期

`GPUModelRunner.execute_model()` 开始时，如果有 KV transfer group，会先处理 preemption / evicted blocks：

```python
if has_kv_transfer_group():
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4075` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4078`

真正 forward 时，ModelRunner 用 connector context 包住模型执行：

```python
with (
    set_forward_context(...),
    self.maybe_get_kv_connector_output(
        scheduler_output,
        defer_finalize=defer_kv_connector_finalize,
    ) as kv_connector_output,
):
    model_output = self._model_forward(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4302` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4326`

如果本轮没有 token 需要 forward，但仍有 KV transfer 要推进：

```python
return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4109` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4112`

这说明：

```text
KV save/load 可以在 0-token step 中推进；
execute_model 不一定等于模型 forward，它也承担 connector transfer 驱动。
```

### 10.3 KVConnectorModelRunnerMixin 的 context 做什么

入口：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:78`

核心流程：

```python
kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
kv_connector.start_load_kv(get_forward_context())
try:
    yield output
finally:
    if wait_for_save and not defer_finalize:
        kv_connector.wait_for_save()

    output.finished_sending, output.finished_recving = (
        kv_connector.get_finished(scheduler_output.finished_req_ids)
    )
    output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
    output.kv_connector_stats = kv_connector.get_kv_connector_stats()
    output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()
    output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()
    kv_connector.clear_connector_metadata()
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:83` 到 `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:112`

这里几个点很重要：

```text
1. start_load_kv() 名字偏 load，但很多 connector 也借这个 metadata 入口推进 save/send；
2. save_kv_layer() 通常发生在 attention layer forward 过程中；
3. wait_for_save() 在 context 退出时保证必要保存完成，避免 KV buffer 被覆盖；
4. get_finished(finished_req_ids) 产出 finished_sending / finished_recving；
5. 这些结果进入 ModelRunnerOutput.kv_connector_output。
```

### 10.4 save_kv_layer 在 attention 层之后调用

attention 层的 KV transfer wrapper 在 forward 后调用：

```python
result = func(*args, **kwargs)
connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
return result
```

位置：`code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:52` 到 `code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:59`

也就是说：

```text
逐层 save 的时间点在该 attention layer 计算完、KV 已经写入 paged KV cache 之后。
```

---

## 11. finished_sending 如何回到 Scheduler

### 11.1 Worker 输出 KVConnectorOutput

`KVConnectorOutput` 定义：

```python
@dataclass
class KVConnectorOutput:
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    kv_connector_stats: KVConnectorStats | None = None
    kv_cache_events: KVConnectorKVEvents | None = None
    kv_connector_worker_meta: KVConnectorWorkerMetadata | None = None
    invalid_block_ids: set[int] = field(default_factory=set)
```

位置：`code/vllm/vllm/v1/outputs.py:195` 到 `code/vllm/vllm/v1/outputs.py:212`

这个对象会挂在 `ModelRunnerOutput.kv_connector_output` 上返回 Scheduler。

### 11.2 Scheduler.update_from_output 消费 kv_connector_output

`update_from_output()` 里取出：

```python
kv_connector_output = model_runner_output.kv_connector_output
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1474`

最后更新 connector 状态：

```python
if kv_connector_output:
    self._update_from_kv_xfer_finished(kv_connector_output)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1731` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1733`

### 11.3 _update_from_kv_xfer_finished 释放 delayed blocks

入口：`code/vllm/vllm/v1/core/sched/scheduler.py:2417`

先让 Scheduler-side connector 消费 Worker 输出：

```python
if self.connector is not None:
    self.connector.update_connector_output(kv_connector_output)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2428` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2429`

然后处理 finished_recving：

```python
for req_id in kv_connector_output.finished_recving or ():
    ...
    if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        self.finished_recving_kv_req_ids.add(req_id)
    else:
        assert RequestStatus.is_finished(req.status)
        self._free_blocks(self.requests[req_id])
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2431` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2440`

再处理 finished_sending：

```python
for req_id in kv_connector_output.finished_sending or ():
    logger.debug("Finished sending KV transfer for request %s", req_id)
    assert req_id in self.requests
    self._free_blocks(self.requests[req_id])
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2441` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2444`

所以：

```text
finished_sending 是释放 request blocks 的最终信号；
它必须对应之前 request_finished*() 返回 delay_free_blocks=True 的请求。
```

---

## 12. NIXL push 模式：请求结束后真正异步 WRITE

NIXL push 是最符合“请求结束后 KV 保存/发送”的例子。

### 12.1 facade 转发 request_finished_all_groups

`NixlBaseConnector` 支持 HMA：

```python
class NixlBaseConnector(KVConnectorBase_V1, SupportsHMA):
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:79`

`request_finished_all_groups()` 直接转发给 scheduler 侧实现：

```python
def request_finished_all_groups(
    self,
    request: Request,
    block_ids: tuple[list[int], ...],
) -> tuple[bool, dict[str, Any] | None]:
    assert self.connector_scheduler is not None
    return self.connector_scheduler.request_finished(request, block_ids)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:202` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:208`

### 12.2 Push Scheduler 在 request_finished 中记录 finished blocks

入口：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py:203`

关键逻辑：

```python
if not params:
    return False, None

is_p_node = bool(params.get("do_remote_decode"))
...
if not is_p_node:
    return False, None

if request.status not in (
    RequestStatus.FINISHED_LENGTH_CAPPED,
    RequestStatus.FINISHED_STOPPED,
):
    ...
    return False, None
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py:211` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py:253`

也就是说：

```text
NIXL push 只在 P-side、且请求正常 stopped / length capped 时保存；
abort / error 等状态通常不保存。
```

如果有 blocks：

```python
delay_free_blocks = any(len(group) > 0 for group in block_ids)
...
self._reqs_need_send[request.request_id] = (
    time.perf_counter() + self._kv_lease_duration
)

block_ids = self.get_sw_clipped_blocks(block_ids)
remote_num_tokens = request.num_computed_tokens
self._finished_request_blocks[request.request_id] = block_ids
self._newly_finished_push_blocks[request.request_id] = block_ids
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py:254` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py:274`

返回：

```python
return delay_free_blocks, dict(
    do_remote_prefill=True,
    do_remote_decode=False,
    remote_block_ids=block_ids,
    remote_engine_id=self.engine_id,
    remote_request_id=request.request_id,
    remote_host=self.side_channel_host,
    remote_port=self.side_channel_port,
    tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
    remote_num_tokens=remote_num_tokens,
)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py:275` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py:285`

这个返回值有两层意义：

```text
delay_free_blocks=True：P 侧 blocks 不能释放，等待 WRITE 完成；
kv_transfer_params：告诉上层/远端 decode 请求，后续可以从这个 P 节点获取 KV。
```

### 12.3 build_connector_meta 把 finished blocks 交给 Worker

```python
if self._newly_finished_push_blocks:
    meta.push_finished_blocks = dict(self._newly_finished_push_blocks)
    self._newly_finished_push_blocks.clear()
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py:327` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py:329`

同时：

```python
def has_pending_push_work(self) -> bool:
    return bool(self._finished_request_blocks or self._push_pending_registrations)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py:333` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py:338`

Scheduler 的 `has_requests()` 会检查这个状态：

```python
return (
    self.has_unfinished_requests()
    or self.has_finished_requests()
    or (self.connector is not None and self.connector.has_pending_push_work())
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2130` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2141`

含义：

```text
即使没有正常请求可跑，只要 push save 还没完成，engine loop 也要继续 step；
否则 Worker 没机会收到 metadata / 轮询完成状态。
```

### 12.4 Push Worker 用 writer thread 执行 WRITE

`NixlPushConnectorWorker.start_load_kv()` 会处理 `metadata.push_finished_blocks`：

```python
if metadata.push_finished_blocks:
    for req_id, block_ids in metadata.push_finished_blocks.items():
        self._finished_blocks_inbox.put((req_id, block_ids))
    self._push_writer_wake.set()
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py:171` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py:175`

writer thread 会匹配 D 侧 registration，并开始 WRITE：

```python
if matched is not None:
    self._do_start_push_kv(rid, blocks, matched)
else:
    self._push_finished_blocks[rid] = blocks
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py:205` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py:216`

真正提交 WRITE：

```python
handle = self.nixl_wrapper.make_prepped_xfer(
    "WRITE",
    local_xfer_side_handle,
    local_block_descs_ids,
    remote_xfer_side_handle,
    remote_block_descs_ids,
    notif_msg=notif_id,
)
self.nixl_wrapper.transfer(handle)
with self._sending_transfers_lock:
    self._sending_transfers[request_id].append(handle)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py:629` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py:642`

### 12.5 Push Worker 汇报 finished_sending

`get_finished()` 中：

```python
done_sending, done_recving = super().get_finished()
...
done_pushing = self._pop_done_transfers(self._sending_transfers)
for req_id in done_pushing:
    ...
    done_sending.add(req_id)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py:716` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py:733`

这最终进入：

```text
KVConnectorOutput.finished_sending
  → Scheduler._update_from_kv_xfer_finished()
  → Scheduler._free_blocks()
```

---

## 13. LMCache：connector 自己决定是否异步保存

LMCache connector facade 的 request_finished：

```python
def request_finished(
    self,
    request: Request,
    block_ids: list[int],
) -> tuple[bool, dict[str, Any] | None]:
    return self._lmcache_engine.request_finished(request, block_ids)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:325` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:340`

它把具体策略交给 LMCache engine：

```text
是否需要保存；
保存哪些 block；
是否异步；
是否返回 kv_transfer_params；
是否需要 delay_free_blocks。
```

Worker 侧逐层保存也是 facade 转发：

```python
self._lmcache_engine.save_kv_layer(
    layer_name, kv_layer, attn_metadata, **kwargs
)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:166` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:187`

保存等待：

```python
self._lmcache_engine.wait_for_save()
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:189` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:197`

完成状态：

```python
return self._lmcache_engine.get_finished(finished_req_ids)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:199` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:213`

所以 LMCache 路径的心智模型是：

```text
vLLM 提供 request_finished/save_kv_layer/wait_for_save/get_finished 框架；
具体保存策略由 LMCache engine 决定。
```

---

## 14. Offloading connector：store job 不一定用 finished_sending

Offloading connector 比 NIXL push 特殊：它更多是“执行过程中逐步 store”，并通过 worker meta 的 completed jobs 更新 Scheduler-side 状态。

### 14.1 build_connector_meta 构造 store jobs

`OffloadingConnectorScheduler.build_connector_meta()`：

```python
meta = OffloadingConnectorMetadata(
    load_jobs=self._current_batch_load_jobs,
    store_jobs=self._build_store_jobs(scheduler_output),
    jobs_to_flush=self._current_batch_jobs_to_flush,
)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1027` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1031`

`_build_store_jobs()` 会根据本轮 scheduled tokens、block hash / offload keys、sliding window / EAGLE 等规则决定哪些 blocks 要 store。

入口：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:812`

### 14.2 Worker 侧提交 store job

`OffloadingConnectorWorker.prepare_store_kv()`：

```python
for job_id, entry in metadata.store_jobs.items():
    self._unsubmitted_store_jobs.append((job_id, entry.transfer_spec))
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:245` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:250`

注释说明：

```text
defer the store to the beginning of the next engine step,
so that offloading starts AFTER transfers related to token sampling,
thereby avoiding delays to token generation.
```

也就是说，Offloading store 可能被延迟到下一步开始提交。

### 14.3 Offloading get_finished 不用 finished_sending 表示 store 完成

`OffloadingConnectorWorker.get_finished()` 注释：

```text
Stores never emit finished_sending — the scheduler tracks store completion
via kv_connector_worker_meta.completed_jobs and fences any block reuse via jobs_to_flush.
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:252` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:260`

它返回：

```python
return set(), finished_recving
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:286`

完成 job 通过：

```python
return meta
```

即 `build_connector_worker_meta()` 返回 `OffloadingWorkerMetadata`。

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:288` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:294`

### 14.4 Scheduler 侧用 completed_jobs 更新 store 状态

`OffloadingConnectorScheduler.update_connector_output()` 读取：

```python
meta = connector_output.kv_connector_worker_meta
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1037` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1048`

遍历 completed jobs：

```python
for job_id, count in meta.completed_jobs.items():
    ...
    if job_status.is_store:
        self.manager.complete_store(job_status.keys, req_status.req_context)
    else:
        self.manager.complete_load(job_status.keys, req_status.req_context)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1082` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1101`

请求结束时：

```python
if not req_status.transfer_jobs:
    del self._req_status[request.request_id]
    return False, None
...
return False, None
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1133` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1167`

注意它返回 False：

```text
Offloading connector 请求结束时通常不通过 delay_free_blocks 保留整个 request；
它用 pending jobs + block_id_to_pending_jobs + jobs_to_flush 保护 block 复用安全。
```

这和 NIXL push 的 `finished_sending → _free_blocks()` 模式不同。

---

## 15. block 释放的三层安全机制

external KV save 里“什么时候能释放 block”有三层机制，容易混淆。

### 15.1 connector_delay_free_blocks

来自：

```text
connector.request_finished*() 返回 True
```

作用：

```text
整个请求的 KV blocks 暂不释放；
等 finished_sending / finished_recving 回来后再 _free_blocks()。
```

典型：

```text
NIXL push P-side WRITE。
```

### 15.2 defer_block_free

Scheduler 初始化时，如果 KV consumer + multiple inflight batches，可能启用：

```python
multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
    self.defer_block_free = True
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:145` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:151`

`_free_request_blocks()` 会根据 in-flight step fence 决定是否真正归还 BlockPool：

```python
if not self.defer_block_free or request.last_sched_seq <= self.processed_step_seq:
    self.kv_cache_manager.free(request)
    return
blocks = self.kv_cache_manager.pop_blocks_for_free(request)
if blocks:
    self.deferred_frees.append((self.sched_step_seq, blocks))
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2077` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2090`

作用：

```text
防止异步/流水并发下，某个 GPU step 还在写 block，但 Scheduler 已经把 block 还给 BlockPool。
```

这和 connector save 延迟释放是不同层面的保护。

### 15.3 connector jobs_to_flush / block_id_to_pending_jobs

Offloading connector 用：

```text
_block_id_to_pending_jobs
_current_batch_jobs_to_flush
jobs_to_flush
```

在 block 可能复用前强制 flush 对应 store jobs。

相关代码：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1005` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1017`

作用：

```text
不是阻止 request 释放；
而是当 block 将被重新分配时，确保还没完成的 store job 先完成。
```

---

## 16. KV save 和 prefix cache / sliding window 的关系

### 16.1 保存前先 remove_skipped_blocks

`_connector_finished()` 在交给 connector 前先调用：

```python
self.kv_cache_manager.remove_skipped_blocks(
    request_id=request.request_id,
    total_computed_tokens=request.num_computed_tokens,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2311` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2316`

这保证 connector 看到的 block table 已经反映当前 attention 类型的保留窗口。

### 16.2 NIXL push 会再做 sliding window clip

NIXL base scheduler 有：

```python
def get_sw_clipped_blocks(self, block_ids: BlockIds) -> BlockIds:
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:219`

注释说明：

```text
Clip the number of blocks to the sliding window size for each kv cache group that employs SWA.
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:219` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:225`

Push request_finished 中也调用：

```python
block_ids = self.get_sw_clipped_blocks(block_ids)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py:267`

### 16.3 Offloading store 会用 block hash / offload key

Offloading Scheduler 会根据 `request.block_hashes` 生成 offload keys，并决定哪些 block 需要 store。

例如 `RequestOffloadState.update_offload_keys()`：

```python
for req_block_hash in islice(... self.req.block_hashes ...):
    group_state.offload_keys.append(
        make_offload_key(req_block_hash, group_config.group_idx)
    )
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:258` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:272`

所以 external KV save 和 prefix cache 的共同基础是：

```text
full block hash / block id / KV group id 三者的对应关系。
```

---

## 17. 常见例子

### 17.1 没有 connector

```text
request stopped
  → _free_request()
  → _connector_finished() 返回 (False, None)
  → _free_blocks()
  → kv_cache_manager.free(request)
  → del self.requests[req_id]
```

没有 external KV save。

### 17.2 NIXL push 正常保存

```text
P-side request stopped
  → _connector_finished()
  → NixlPushConnectorScheduler.request_finished()
  → delay_free_blocks=True
  → Scheduler 不释放 blocks
  → build_connector_meta() 下发 push_finished_blocks
  → P Worker writer thread 等 D registration
  → NIXL WRITE P blocks 到 D blocks
  → Worker get_finished() 返回 finished_sending={req_id}
  → Scheduler._update_from_kv_xfer_finished()
  → _free_blocks(req)
```

### 17.3 LMCache 保存

```text
request stopped
  → LMCacheConnectorV1.request_finished(request, block_ids)
  → LMCache engine 决定是否保存、是否异步、是否延迟释放
  → Worker 侧 save_kv_layer / wait_for_save / get_finished 按 LMCache 实现推进
  → 如果返回 finished_sending，Scheduler 释放 blocks
```

### 17.4 Offloading store

```text
request running / prefill chunk
  → build_connector_meta()
  → _build_store_jobs()
  → Worker prepare_store_kv()
  → 下一 step 提交 transfer_async
  → Worker meta.completed_jobs 回传
  → Scheduler complete_store()

request finished
  → request_finished() 通知 manager.on_request_finished()
  → 通常返回 delay_free_blocks=False
  → pending store 通过 jobs_to_flush / block_id_to_pending_jobs 保护
```

---

## 18. 容易疑惑的点

### 18.1 请求结束时一定会保存 KV 吗？

不一定。

取决于：

```text
是否配置 KV connector；
request.kv_transfer_params 是否存在；
请求结束状态是否允许保存；
connector 类型和策略；
是否有可保存 blocks；
是否 prompt-only / offload policy / sliding window 过滤后还有 blocks。
```

### 18.2 `request_finished()` 返回 True 表示保存已经完成吗？

不是。

```text
True 表示 connector 接管 blocks，需要异步保存/发送；
完成信号是后续 Worker 返回 finished_sending。
```

### 18.3 `finished_sending` 和 `finished_recving` 有什么区别？

```text
finished_sending：本地 blocks 向外部/远端发送或保存完成，通常用于释放已 finished request 的 blocks；
finished_recving：外部 KV load 到本地完成，通常用于恢复 WAITING_FOR_REMOTE_KVS 请求。
```

在 `_update_from_kv_xfer_finished()` 中：

```text
finished_sending → _free_blocks()
finished_recving → 如果请求还在 WAITING_FOR_REMOTE_KVS，则加入 finished_recving_kv_req_ids；如果请求已经 finished，则 _free_blocks()
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2431` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2444`

### 18.4 Worker 侧为什么在 0-token step 也要执行 connector？

因为可能只有 KV transfer 需要推进，没有模型 token 要 forward。

`kv_connector_no_forward()` 明确注释：

```text
KV send/recv even if no work to do.
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:36` 到 `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:48`

### 18.5 `save_kv_layer()` 和 `request_finished()` 是同一个保存点吗？

不是。

```text
save_kv_layer()：Worker 侧 attention layer forward 后，保存当前层 KV；
request_finished()：Scheduler 侧请求结束时，把 request/block_ids 交给 connector 决策。
```

有的 connector 主要依赖 `save_kv_layer()`，有的 connector 主要依赖 `request_finished()` 生成 finished blocks / lease / transfer params。

### 18.6 为什么 `_connector_finished()` 要先 remove_skipped_blocks？

因为 sliding window / local attention / Mamba 等场景下，有些历史 blocks 对后续 attention 已经不可达。

```text
先清理 skipped blocks，可以避免 connector 保存不需要或不正确的 block 范围。
```

### 18.7 为什么 Offloading store 不走 finished_sending？

Offloading 的 store job 是持续的后台 offload，不一定和请求结束一一对应。

它通过：

```text
kv_connector_worker_meta.completed_jobs
jobs_to_flush
_block_id_to_pending_jobs
```

来确保外部 store 完成和 block 复用安全，而不是用 `finished_sending` 释放整个 request。

---

## 19. 总结

请求结束后 external KV save 的核心链路是：

```text
request finished
  → Scheduler._free_request()
  → Scheduler._connector_finished()
      → remove_skipped_blocks()
      → get_block_ids()
      → connector.request_finished*()
  → connector_delay_free_blocks ? keep request/blocks : free immediately
  → build_connector_meta()
  → Worker connector 执行 save/send
  → KVConnectorOutput.finished_sending
  → Scheduler._update_from_kv_xfer_finished()
  → _free_blocks()
```

最关键的边界是：

```text
Scheduler 负责决定“请求结束时 blocks 能不能释放”；
Connector 负责决定“这些 blocks 是否需要保存/发送，以及是否异步”；
Worker 负责真正执行 KV save/send 并回传完成状态；
finished_sending 是 delayed-free 请求最终释放 blocks 的信号。
```

如果只记一句话：

```text
external KV save 是一套跨 Scheduler 和 Worker 的延迟释放协议：请求结束时 connector 可以接管 KV blocks，Scheduler 暂不释放，Worker 完成 save/send 后用 finished_sending 通知 Scheduler 归还 blocks。
```
