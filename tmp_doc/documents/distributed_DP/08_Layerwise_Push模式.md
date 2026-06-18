# 第 8 章：Layerwise Push 模式

标准 `MooncakeConnectorV1` 是 **D 侧 pull**：P 完成 prefill 后，D 从 P 拉 KV。`MooncakeLayerwiseConnector` 则是 **P 侧 push**：D 先预分配 blocks 并把自己的地址暴露出来，P 在 prefill 的每一层执行过程中把当前层 KV 直接写到 D 节点。

代码位置：`code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`。

## 8.1 两种模式对比

| 维度 | Pull 模式 `MooncakeConnectorV1` | Push 模式 `MooncakeLayerwiseConnector` |
|------|----------------------------------|----------------------------------------|
| 传输方向 | D 从 P 读 | P 向 D 写 |
| Mooncake API | `batch_transfer_sync_read` | `batch_transfer_sync_write` |
| 触发时机 | P prefill 完成后，D 开始拉取 | P prefill 过程中逐层触发 |
| side-channel | P 侧监听 `GET_META_MSG` / `DONE_RECVING_MSG` | D 侧监听 `GET_META_MSG` / `DONE_SENDING_MSG` / `FAILED_SENDING_MSG` |
| 完成条件 | D 拉完后通知 P 释放 blocks | P 每层发送，最后一层完成后通知 D 可 decode |
| 适用目标 | 简单稳定的远端 KV 拉取 | 通信与 prefill 计算重叠，降低等待 |

## 8.2 顶层类

```python
class MooncakeLayerwiseConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeLayerwiseConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeLayerwiseConnectorScheduler(
                vllm_config, kv_cache_config, str(self.engine_id)
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = MooncakeLayerwiseConnectorWorker(
                vllm_config, kv_cache_config, str(self.engine_id)
            )
```

Worker 端与 attention layer 集成：

```python
def start_load_kv(self, forward_context, **kwargs):
    self.connector_worker.start_load_kv(self._connector_metadata)

def wait_for_layer_load(self, layer_name):
    self.connector_worker.wait_for_layer_load(layer_name)

def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
    self.connector_worker.save_kv_layer(
        layer_name, kv_layer, attn_metadata, self._connector_metadata
    )
```

`attention/utils.py` 中会在 attention 逻辑里调用：

```python
connector.wait_for_layer_load(layer_name)
connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)
```

## 8.3 D 侧调度：先发 metaserver 请求

D 侧收到 `do_remote_prefill=True` 请求后：

```python
def update_state_after_alloc(self, request, blocks, num_external_tokens):
    if params is not None and params.get("do_remote_prefill"):
        local_block_ids = blocks.get_block_ids() if num_external_tokens > 0 else []
        remote_block_ids = self._trim_hybrid_remote_block_ids(local_block_ids, len(request.prompt_token_ids))
        remote_cached_tokens = request.num_computed_tokens

        self._reqs_need_recv[request.request_id] = (request, [], local_block_ids)
        params["do_remote_prefill"] = False

        kv_transfer_params = dict(
            request_id=get_external_request_id(request.request_id),
            do_remote_prefill=False,
            do_remote_decode=True,
            remote_block_ids=remote_block_ids,
            remote_block_size=self.block_size,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_tp_size=tensor_parallel_size,
            remote_pcp_size=prefill_context_parallel_size,
            remote_dcp_size=decode_context_parallel_size,
            remote_cached_tokens=remote_cached_tokens,
        )
        self.executor.submit(self._access_metaserver, url=params.get("metaserver"), message=kv_transfer_params)
```

含义：

- D 侧先分配好本地 blocks。
- D 侧把自己的 block ids、host、port、engine_id 通过 metaserver 告诉 Proxy/P 侧。
- P 侧收到后以 `do_remote_decode=True` 执行 prefill，并在每层把 KV 推回 D。

## 8.4 D 侧 Worker：监听接收完成

D Worker 注册 KV Cache 后启动 `KVCacheRecvingLayerThread`：

```python
if kv_transfer_config.is_kv_consumer:
    self.kv_recv_layer_thread = KVCacheRecvingLayerThread(
        self.tp_rank,
        self.side_channel_port,
        self.tp_size,
        self.pd_head_ratio,
        self.engine_id,
        metadata,
        ready_event,
    )
    self.kv_recv_layer_thread.start()
```

监听端口：

```python
handshake_port = self.side_channel_port + self.tp_rank
path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
```

处理消息：

| 消息 | 发送方 | 作用 |
|------|--------|------|
| `GET_META_MSG` | P 侧发送线程 | 获取 D 侧 layer metadata 和 TransferEngine rpc port |
| `DONE_SENDING_MSG` | P 侧发送线程 | 当前请求的 KV 已推送完成 |
| `FAILED_SENDING_MSG` | P 侧发送线程 | 当前请求 KV 推送失败 |

D 侧 `get_finished()` 会把 `DONE_SENDING_MSG` 中的 external request id 映射回本地 request id：

```python
done_recving = self.kv_recv_layer_thread.get_and_clear_done_requests()
done_recving = {self.request_map[s] for s in done_recving if s in self.request_map}
```

## 8.5 P 侧调度：记录待发送请求

P 侧收到 metaserver 转发的请求，`kv_transfer_params` 中有 `do_remote_decode=True`。

```python
if params is not None and params.get("do_remote_decode"):
    local_block_ids = list(blocks.get_block_ids())
    remote_cache_tokens = params["remote_cached_tokens"]
    self._reqs_need_send_layerwise[request.request_id] = SendReqInfo(
        local_block_ids=local_block_ids,
        local_transferred_tokens=remote_cache_tokens,
        local_computed_tokens=0,
        request=request,
    )
```

随后 `build_connector_meta()` 在每个 scheduler step 更新：

- 新分配的 blocks
- 当前已计算 tokens
- 当前已发送 tokens
- chunk 是否完成

```python
send_req_info.update_transferred_tokens(
    round_down(send_req_info.local_computed_tokens, min(self.block_size))
)
send_req_info.update_computed_tokens(
    computed_tokens + scheduled_tokens - spec_decode_tokens
)
chunk_finish = send_req_info.local_computed_tokens >= len(request.all_token_ids)
meta.add_new_req(..., chunk_finish=chunk_finish)
```

## 8.6 P Worker start_load_kv：计算发送映射

P 侧 `start_load_kv()` 不是真的 load，而是把 scheduler metadata 转成发送任务所需的 remote/local 映射：

```python
elif kv_transfer_config.is_kv_producer:
    update_metadata = {}
    for req_idx, (req_id, req_meta) in enumerate(metadata.requests.items()):
        transfer_mappings = {}
        self._align_remote_block_ids(req_meta)
        for i, kv_cache_spec in enumerate(self.kv_cache_specs):
            if isinstance(kv_cache_spec, MambaSpec):
                single_group_transfer_mappings = self._get_kv_split_metadata_for_mamba(...)
            else:
                single_group_transfer_mappings = self._get_kv_split_metadata(...)
            ...
        update_req_meta.local_block_ids = self._get_kernel_block_ids(block_dict["local_block_ids"])
        update_req_meta.remote_block_ids = self._get_kernel_block_ids(block_dict["remote_block_ids"])
        update_req_meta.trans_count = block_dict["trans_count"]
```

P 侧 push 比 pull 更复杂，因为需要边计算边传输：

- `local_computed_tokens`：P 当前已算到哪里。
- `local_transed_tokens`：上一次已经传到哪里。
- `chunk_finish`：当前请求是否已经完成所有 prompt/chunk。
- `trans_count`：D 侧需要收到多少个发送完成信号。

## 8.7 每层 save_kv_layer

P attention 每层执行后调用：

```python
def save_kv_layer(self, layer_name, kv_layer, attn_metadata, connector_metadata):
    if is_kv_producer and connector_metadata.requests.keys():
        reshape_cache_event = attn_metadata.reshape_cache_event or torch.npu.Event()
        send_task = connector_metadata.send_task
        layer_group_idx = self.layer_metadata[layer_name].tensor_group_idx[0]
        ...
        layer_send_task = SendTask(
            wait_event=reshape_cache_event,
            k_cache=keys,
            v_cache=values,
            layer_idx=self.current_layer,
            layer_name=layer_name,
            group_rearrange_block_ids=send_task.group_rearrange_block_ids,
        )
        layer_send_task.send_request[req_id] = req_meta_update
        self.kv_send_layer_thread.send_queue.put(layer_send_task)
        self.current_layer += 1
```

如果需要 reshard / quant / NZ：

- `pd_head_ratio != 1`：先从 paged cache load 到临时 buffer，然后 all-to-all 重排。
- `enable_c8_quant`：生成 int8 quant KV。
- `enable_kv_quant`：使用 vLLM quantize，再转换 NZ。

## 8.8 KVCacheSendingLayerThread

发送线程处理每层任务：

```python
class KVCacheSendingLayerThread(threading.Thread):
    def run(self):
        torch.npu.set_device(local_rank)
        self.ready_event.set()
        while True:
            send_task = self.send_queue.get()
            self._handle_request(send_task)
```

核心 `_transfer_kv_cache()`：

1. 如果需要重排，等待 `resharding_stream`。
2. 如果不需要重排，等待 `reshape_cache_event.synchronize()`。
3. 按 session 合并传输任务。
4. 调用 Mooncake `batch_transfer_sync_write()`。
5. 最后一层且 chunk 完成时，通知 D 侧完成。

```python
ret = self.engine.batch_transfer_sync_write(
    session_id,
    transfer_meta.src,
    transfer_meta.dst,
    transfer_meta.length,
)
```

## 8.9 地址计算

对于普通 Attention 且 `pd_head_ratio == 1`：

```python
src = src_layer_base_addr + group_local_block_id[0] * block_len
dst = dst_layer_base_addr + group_remote_block_id[0] * block_len
length = len(group_local_block_id) * block_len
```

对于 `pd_head_ratio > 1`：

```python
src = k_buffer.data_ptr() + rearranged_local_block_idx * block_len
dst = remote_base_addr + remote_block_id * remote_block_len \
      + block_len * ((tp_rank // num_head_replica) % pd_head_ratio)
length = block_len
```

Mamba state 会分别处理 conv / ssm，并在 TP ratio > 1 时按 head/state 维度切片。

## 8.10 P 侧查询 D metadata

P 侧在第一次给某个 D 端口发送前，会查询 D 侧 metadata：

```python
encoded_data = self.encoder.encode((GET_META_MSG, req_id))
sock = self._get_remote_socket(req_meta.remote_host, req_meta.remote_port)
ensure_zmq_send(sock, encoded_data, path)
metadata_bytes = ensure_zmq_recv(sock, self.remote_poller, path)
agent_meta = self.decoder.decode(metadata_bytes)

self.remote_layer_metadata[remote_engine_id][remote_port] = agent_meta.layer_metadata
self.remote_te_port[remote_engine_id][remote_port] = agent_meta.te_rpc_port
```

这样 P 侧知道 D 侧每层 KV Cache 的地址和 D 侧 TransferEngine rpc port。

## 8.11 完成通知

最后一层传输完成时：

```python
if send_task.layer_idx == (self.total_layers - 1):
    for req_id in transfer_meta.req_ids:
        if req_meta.chunk_finish:
            self.callback_func(req_id, req_meta, layer_group_idx, trans_flag=True)
```

`callback_func` 是 `send_done_send_signal()`：

```python
send_msg_type = DONE_SENDING_MSG if trans_flag else FAILED_SENDING_MSG
encoded_data = msgpack.encode((send_msg_type, external_req_id, req_meta.trans_count[group_idx], side_channel_path))
ensure_zmq_send(sock, encoded_data, f"{remote_host}:{remote_port}")
```

D 侧 `KVCacheRecvingLayerThread` 收到后：

```python
elif msg[0] == DONE_SENDING_MSG:
    self.update_done_task(request_id, trans_count, side_channel_path)
    sock.send_multipart((identity, b"", b"ACK"))
elif msg[0] == FAILED_SENDING_MSG:
    self.update_failed_task(request_id)
    sock.send_multipart((identity, b"", b"ACK"))
```

## 8.12 Layerwise 全链路时序

```text
D 请求到达
    │
    ├── D Scheduler: do_remote_prefill=True
    ├── D 分配本地 blocks
    ├── D 通过 metaserver 通知 P：remote_block_ids / remote_host / remote_port
    └── D Worker: KVCacheRecvingLayerThread 监听 GET_META / DONE_SENDING

P 收到 metaserver 转发请求
    │
    ├── P Scheduler: do_remote_decode=True
    ├── P build_connector_meta: 记录 SendReqInfo
    ├── P Worker start_load_kv: 计算每个 group 的传输映射
    │
    ├── P attention layer 0 完成
    │       └── save_kv_layer → KVCacheSendingLayerThread → batch_transfer_sync_write
    ├── P attention layer 1 完成
    │       └── save_kv_layer → batch_transfer_sync_write
    ├── ...
    └── P 最后一层完成
            └── send_done_send_signal(DONE_SENDING_MSG)

D 收到 DONE_SENDING_MSG
    │
    ├── get_finished() 返回 done_recving
    └── 请求离开 WAITING_FOR_REMOTE_KVS，继续 decode
```

## 8.13 Layerwise 模式特点

- `request_finished()` 不再负责延迟释放 blocks，代码直接返回 `(False, None)`。
- 传输完成由 P 侧显式 `DONE_SENDING_MSG` 通知 D。
- 传输可以与 P 侧 prefill 层计算重叠。
- D 侧 `wait_for_layer_load()` 当前实现为空，实际等待主要依赖请求级 `WAITING_FOR_REMOTE_KVS` 状态和 DONE 信号。
- 非对称 TP、quant、NZ、attn+mamba hybrid 下会引入额外 buffer 和 resharding stream。
