# 第 6 章：Prefill 端延迟释放与完成确认

在 P/D 分离 pull 模式中，P 节点完成 prefill 后不能立刻释放 KV Cache blocks。原因是 D 节点还没有把这些 KV Cache 拉到自己的本地 KV buffer 中。

因此 P 侧需要两套机制：

1. Scheduler 端 `request_finished()` 返回 `delay_free_blocks=True`，让 vLLM 暂缓释放 blocks。
2. Worker 端 `KVCacheSendingThread` 监听 D 侧完成信号，完成后通过 `get_finished()` 告诉 Scheduler 可以释放。

## 6.1 P 侧 prefill 请求的生命周期

```text
Proxy → P 节点请求
    │
    ├── kv_transfer_params.do_remote_decode = True
    ├── max_tokens / max_completion_tokens = 1
    │
P Scheduler 调度
    │
    ├── get_num_new_matched_tokens()
    │       └── 如果需要 state 截断，则删除 prompt 最后一个 token
    │
P Worker 执行 prefill
    │
    └── KV Cache 写入 P 本地 NPU KV buffer

请求结束
    │
    └── request_finished()
            ├── 计算可传输 remote_block_ids
            ├── 返回 kv_transfer_params 给 Proxy / D 节点
            ├── delay_free_blocks = True
            └── _reqs_need_send[request_id] = time.time()
```

## 6.2 request_finished 的触发条件

```python
if (
    params is None
    or not params.get("do_remote_decode")
    or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
):
    return False, None
```

只有满足以下条件，P 侧才会返回 remote KV 信息：

- 请求带有 `kv_transfer_params`
- `do_remote_decode=True`
- 请求状态是 `FINISHED_LENGTH_CAPPED`

为什么要求 `FINISHED_LENGTH_CAPPED`？

P 侧通常被 Proxy 设置为 `max_tokens=1` 或 `max_completion_tokens=1`，它的目标不是完整生成，而是完成 prefill 并产出最小输出，使请求以长度截断结束，从而触发 remote KV 返回。

## 6.3 计算 remote_block_ids

P 侧不是把所有 block 都交给 D 侧，而是只返回真正包含 prompt KV 的 blocks。

```python
num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)
computed_block_ids = self._get_transfer_block_ids(block_ids, len(request.prompt_token_ids))
computed_block_ids = self._get_swa_transfer_block_ids(computed_block_ids)
```

### 6.3.1 _get_transfer_block_ids

```python
def _get_transfer_block_ids(self, block_ids, prompt_len):
    transfer_block_ids = []
    for blocks, group_info in zip(block_ids, self.group_transfer_info):
        if group_info.is_state_group:
            transfer_block_ids.append(blocks)
        else:
            num_prompt_blocks = cdiv(prompt_len, group_info.tokens_per_block)
            transfer_block_ids.append(blocks[:num_prompt_blocks])
    return tuple(transfer_block_ids)
```

含义：

- Attention-like group：按 prompt token 数推导需要多少 blocks。
- Compress group：`tokens_per_block = block_size * compress_ratio`。
- State group（如 Mamba）：不是普通 context block 对齐，保留原 blocks。

### 6.3.2 _get_swa_transfer_block_ids

```python
def _get_swa_transfer_block_ids(self, block_ids):
    for blocks, group_info in zip(block_ids, self.group_transfer_info):
        if group_info.is_state_group or group_info.blocks_per_window == 0:
            transfer_block_ids.append(blocks)
        else:
            window_blocks = blocks[-group_info.blocks_per_window:]
            transfer_block_ids.append([block_id for block_id in window_blocks if block_id != 0])
```

SlidingWindow group 只传 window 内的尾部 blocks，并过滤 placeholder block 0。

## 6.4 返回给 D 侧的 kv_transfer_params

`request_finished()` 返回的 dict 会通过 Proxy 进入 D 请求：

```python
return delay_free_blocks, dict(
    do_remote_prefill=True,
    do_remote_decode=False,
    remote_block_ids=computed_block_ids,
    remote_engine_id=self.engine_id,
    remote_request_id=request.request_id,
    remote_host=self.side_channel_host,
    remote_port=self.side_channel_port,
    remote_pcp_size=self.pcp_size,
    remote_dcp_size=self.dcp_size,
    remote_ptp_size=self.tp_size,
    last_token_id=request.output_token_ids[-1],
    remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
    num_prompt_blocks=num_prompt_blocks,
)
```

字段说明：

| 字段 | 作用 |
|------|------|
| `do_remote_prefill=True` | 告诉 D 侧需要 remote KV prefill |
| `remote_block_ids` | P 侧可被读取的 block ids |
| `remote_engine_id` | P engine id，用于远端 metadata cache |
| `remote_request_id` | P 侧 request id，用于 DONE 信号和 rank 选择 hash |
| `remote_host` / `remote_port` | P 侧 side-channel 基址 |
| `remote_pcp_size` / `remote_dcp_size` / `remote_ptp_size` | P 侧并行配置 |
| `last_token_id` | P 侧最小生成输出 token，可供上层协议使用 |
| `remote_multi_nodes_meta_mapping` | 跨节点 rank → host/engine_id |
| `num_prompt_blocks` | P prompt 总 blocks，用于 CP/prefix cache 映射 |

## 6.5 _reqs_need_send 与 requests_to_send

P 侧 `delay_free_blocks=True` 时：

```python
self._reqs_need_send[request.request_id] = time.time()
```

随后 `build_connector_meta()` 把它放进 Worker metadata：

```python
meta.requests_to_send = self._reqs_need_send
self._reqs_need_send = {}
```

Worker 的 `start_load_kv()` 会处理它：

```python
if self.kv_send_thread is not None and self.pcp_size * self.dcp_size == 1:
    for req_id, delay_start_time in metadata.requests_to_send.items():
        if self.tp_rank in self._prefill_get_remote_rank(req_id):
            self.kv_send_thread.add_delayed_request(req_id, delay_start_time)
        else:
            self.kv_send_thread.add_not_transfer_request(req_id)

if self.kv_send_thread is not None and self.pcp_size * self.dcp_size > 1:
    for req_id, delay_start_time in metadata.requests_to_send.items():
        self.kv_send_thread.add_delayed_request(req_id, delay_start_time)
```

含义：

- 只有真正会被 D 侧拉取的 P rank 才进入 delayed request。
- 不会被拉取的 P rank 直接标记完成，避免无意义等待。
- PCP/DCP 场景下可能多个 remote port 都参与，因此所有相关 rank 都延迟。

## 6.6 KVCacheTaskTracker

P 侧和 D 侧线程都使用 `KVCacheTaskTracker` 追踪完成状态：

```python
class KVCacheTaskTracker:
    finished_requests: set[str]
    delayed_free_requests: OrderedDict[str, float]
    reqs_to_process: set[str]
```

核心方法：

```python
def add_req_to_process(self, request_id):
    self.reqs_to_process.add(request_id)

def add_not_transfer_request(self, request_id):
    self.finished_requests.add(request_id)
    self.reqs_to_process.discard(request_id)

def add_delayed_request(self, request_id, delay_start_time):
    if request_id in self.reqs_to_process:
        self.delayed_free_requests[request_id] = delay_start_time

def update_done_task_count(self, request_id):
    if request_id in self.reqs_to_process:
        self.finished_requests.add(request_id)
        self.reqs_to_process.discard(request_id)
        self.delayed_free_requests.pop(request_id, None)
```

## 6.7 D 侧 DONE_RECVING_MSG

D 侧每个拉取任务完成后，会向 P 侧发送：

```python
(DONE_RECVING_MSG, request_id, remote_port_send_num)
```

P 侧 `KVCacheSendingThread.run_busy_loop()` 处理：

```python
elif msg[0] == DONE_RECVING_MSG:
    request_id = msg[1]
    remote_port_send_num = msg[2]
    if remote_port_send_num:
        self.port_send_num[request_id] += 1
        if self.port_send_num[request_id] >= remote_port_send_num[handshake_port]["num"]:
            self.task_tracker.update_done_task_count(request_id)
    else:
        self.task_tracker.update_done_task_count(request_id)
    sock.send_multipart((identity, b"", b"ACK"))
```

`remote_port_send_num` 用于 PCP/DCP、多端口场景：同一个 P port 可能需要等多个 D 侧拉取完成，计数达到要求后才真正释放。

## 6.8 get_finished 释放 blocks

Worker 端：

```python
def get_finished(self):
    done_sending = (
        self.kv_send_thread.get_and_clear_finished_requests()
        if self.kv_role == "kv_producer"
        else set()
    )
    done_recving = (...)
    return done_sending, done_recving
```

Scheduler / engine 根据 `done_sending` 得知：这些请求之前 `request_finished()` 返回了 `delay_free_blocks=True`，现在可以安全释放对应 blocks。

## 6.9 超时兜底

`KVCacheTaskTracker._retrieve_expired_requests()` 会检查延迟释放时间：

```python
if current_time - delay_start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
    self.delayed_free_requests.popitem(last=False)
    self.reqs_to_process.discard(request_id)
    expired_requests.add(request_id)
```

如果 D 侧异常退出或 DONE 信号丢失，P 侧不会无限持有 KV blocks，而是在超时后强制释放，避免显存泄漏。

## 6.10 完整确认链路

```text
P request_finished()
    │
    ├── 返回 delay_free_blocks=True
    ├── 返回 remote_block_ids / host / port / engine_id
    └── _reqs_need_send[req_id] = now

P build_connector_meta()
    │
    └── meta.requests_to_send[req_id] = now

P Worker start_load_kv()
    │
    └── KVCacheSendingThread.task_tracker.add_delayed_request(req_id)

D Worker KVCacheRecvingThread
    │
    ├── batch_transfer_sync_read() 拉取 KV
    └── ZMQ 发送 DONE_RECVING_MSG(req_id)

P KVCacheSendingThread
    │
    ├── 收到 DONE_RECVING_MSG
    ├── update_done_task_count(req_id)
    └── 返回 ACK

P Worker get_finished()
    │
    └── done_sending 包含 req_id

Scheduler / Engine
    │
    └── 释放延迟 blocks
```
