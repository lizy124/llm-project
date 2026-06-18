# 第 5 章：Decode 端拉取 KV Cache（Pull 模式）

本章讲 `MooncakeConnectorV1` 的核心路径：D 节点从 P 节点拉取 KV Cache。

## 5.1 触发入口

D Scheduler 检测到 `do_remote_prefill=True` 后返回 `(count, True)`，请求进入 `WAITING_FOR_REMOTE_KVS`。随后 `build_connector_meta()` 生成 `MooncakeConnectorMetadata`，Worker 在 forward 前调用：

```python
def start_load_kv(self, forward_context, **kwargs):
    self.connector_worker.start_load_kv(self._connector_metadata)
```

Worker 端入口：

```python
def start_load_kv(self, metadata: MooncakeConnectorMetadata):
    for req_id in metadata.reqs_in_batch:
        if self.kv_recv_thread is not None:
            self.kv_recv_thread.task_tracker.add_req_to_process(req_id)

    for req_id, meta in metadata.requests.items():
        remote_req_id = meta.remote_request_id
        prefill_tp_size = meta.remote_ptp_size or self._prefill_tp_size

        remote_handshake_port_list, local_block_ids_list, remote_block_ids_list = \
            self._get_kv_split_metadata(remote_req_id, meta)

        group_pulls_list = self._get_group_pulls_metadata(
            remote_req_id,
            remote_handshake_port_list,
            prefill_tp_size,
            meta.remote_port,
        )

        for pcp_dcp_rank, remote_ports in enumerate(remote_handshake_port_list):
            for remote_tp_offset, remote_handshake_port in enumerate(remote_ports):
                self.kv_recv_thread.add_request(...)
```

## 5.2 任务拆分：remote port / local blocks / remote blocks

`_get_kv_split_metadata()` 负责决定当前 D rank 要从哪些 P rank 拉取哪些 blocks。

返回三个对齐的列表：

| 返回值 | 含义 |
|--------|------|
| `remote_handshake_port_list` | 每个传输 shard 对应的 P 侧握手端口列表 |
| `local_block_ids_list` | D 侧本地写入 blocks，按 KV group 分组 |
| `remote_block_ids_list` | P 侧读取 blocks，按 KV group 分组 |

简单场景（无 PCP/DCP）：

```python
chosen_rank_list = self._get_remote_rank(req_id, prefill_tp_size)
remote_handshake_port_list = [[x + meta.remote_port for x in chosen_rank_list]]
local_block_ids_list = [meta.local_block_ids]
remote_block_ids_list = [meta.remote_block_ids]
```

复杂 PCP/DCP 场景会根据：

- P/D 的 PCP/DCP 大小
- P/D 的 KV head group
- prefix cache 已命中 blocks
- `num_prompt_blocks`
- 当前 D rank 的 `pcp_rank/dcp_rank/tp_rank`

拆出多个 shard。

## 5.3 group_pulls：每个远端 port 拉哪些 KV group

`_get_group_pulls_metadata()` 会把“从哪个 P rank 拉”进一步细化成“拉哪个 KV group、TP offset、PP rank”：

```python
@dataclass(frozen=True)
class GroupPull:
    group_id: int
    remote_tp_offset: int
    num_group_pulls: int
    prefill_pp_rank: int = 0
    is_group_transfer_end: bool = False
```

字段说明：

| 字段 | 含义 |
|------|------|
| `group_id` | KV cache group id |
| `remote_tp_offset` | 当前拉取的是 P 侧 TP shard 中的第几个 offset |
| `num_group_pulls` | 当前 group 需要几个 P shard 拼成 D rank 需要的数据 |
| `prefill_pp_rank` | 该数据来自 P 侧哪个 PP stage |
| `is_group_transfer_end` | 是否是该 group 最后一个 shard，用于决定何时 reformat |

## 5.4 入队到 KVCacheRecvingThread

每个 `(pcp_dcp_rank, remote_tp_offset)` 组合都会入队：

```python
self.kv_recv_thread.add_request(
    request_id=req_id,
    remote_request_id=remote_req_id,
    local_block_ids=local_block_ids_list[pcp_dcp_rank],
    remote_block_ids=remote_block_ids_list[pcp_dcp_rank],
    group_pulls=group_pulls_list[pcp_dcp_rank][remote_tp_offset],
    remote_engine_id=remote_engine_id,
    remote_host=remote_host,
    remote_handshake_port=remote_handshake_port,
    remote_port_send_num=remote_port_send_num,
    num_computed_tokens=meta.num_computed_tokens,
    all_task_done=(最后一个 shard),
)
```

`all_task_done=True` 的最后一个 shard 完成时，接收线程会把请求加入 `finished_requests`，Scheduler 才会认为 remote KV ready。

## 5.5 接收线程处理请求

```python
class KVCacheRecvingThread(threading.Thread):
    def run(self):
        self.ready_event.set()
        while True:
            request_data = self.request_queue.get()
            self._handle_request(request_data)
```

`_handle_request()` 逻辑：

```python
def _handle_request(self, req_meta):
    request_id = req_meta["request_id"]
    remote_request_id = req_meta["remote_request_id"]

    try:
        if transfer_failed:
            self._mark_failed_recv_request(request_id, local_block_ids)
        else:
            self._transfer_kv_cache_all_groups(req_meta)
    except Exception:
        self._mark_failed_recv_request(request_id, local_block_ids)
    finally:
        if all_task_done:
            self.task_tracker.update_done_task_count(request_id)
            self._clear_failed_recv_request(request_id)
        self.request_queue.task_done()
        self._send_done_signal_to_free_remote_port(...)
        self._send_done_recv_signal(...)
```

关键点：

- 即使传输失败，也会发送完成信号，避免 P 侧资源泄漏。
- 失败时会记录 `invalid_block_ids`，供 Scheduler 后续识别需要重试/不使用的 blocks。

## 5.6 获取 P 侧 metadata

第一次从某个 P worker 拉取时，D 侧需要获取 P 侧 metadata：

```python
def _get_remote_metadata(self, remote_host, remote_handshake_port):
    sock = self._get_remote_socket(remote_host, remote_handshake_port)
    ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")), path)
    metadata_bytes = ensure_zmq_recv(sock, self.remote_poller, path)
    agent_meta = self.decoder.decode(metadata_bytes)

    self.remote_kv_group2layeridx[engine_id][remote_handshake_port] = agent_meta.kv_group2layeridx
    self.kv_caches_base_addr[engine_id][remote_handshake_port] = agent_meta.kv_caches_base_addr
    self.remote_te_port[engine_id][remote_handshake_port] = agent_meta.te_rpc_port
    self.remote_block_size_scale[engine_id][remote_handshake_port] = agent_meta.block_size_scale
    self.remote_block_stride_per_addr[engine_id][remote_handshake_port] = agent_meta.block_strides
```

P 侧 `KVCacheSendingThread` 收到 `GET_META_MSG` 后直接返回 encoded `MooncakeAgentMetadata`。

## 5.7 构建 TransferEngine 读任务

`_transfer_kv_cache_all_groups()` 是真正生成 `src_list/dst_list/length_list` 的地方。

核心流程：

```python
remote_kv_caches_base_addrs = self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
local_kv_caches_base_addrs = self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port]
remote_transfer_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
session_id = f"{remote_host}:{remote_transfer_port}"

src_list, dst_list, length_list = [], [], []
for group_pull in group_pulls:
    group_idx = group_pull.group_id
    group_spec, layer_indices = self.kv_group2layeridx[group_idx]
    layer_indices = pp_layer_indices(layer_indices, group_pull.prefill_pp_rank)
    ...
```

对于 Attention-like group：

```python
local_scale = self.block_size_scale[layer_indices[0]][0]
remote_scale = remote_block_size_scale[layer_indices[0]][0]
kernel_local_block_ids = expand_block_ids(local_group_block_ids, local_scale)
kernel_remote_block_ids = expand_block_ids(remote_group_block_ids, remote_scale)

remote_start_idx = num_computed_tokens // remote_kernel_token_size
kernel_remote_block_ids = kernel_remote_block_ids[remote_start_idx:]
num_kernel_blocks = min(len(kernel_remote_block_ids), len(kernel_local_block_ids))
```

这段用于处理：

- P/D 物理 kernel block scale 不同
- D 侧已有 prefix cache，远端只需拉剩余部分
- local/remote block 数不完全一致

## 5.8 地址计算

Attention-like group 每层每个 cache tensor：

```python
src = src_layer_base_addr + local_block_id[0] * block_stride + inner_offset * inner_block_len
dst = dst_layer_base_addr + remote_block_id[0] * remote_block_stride
length = inner_block_len * len(local_block_id)
```

这里变量命名需要结合 Mooncake `batch_transfer_sync_read(session_id, src, dst, len)` 理解：

- `session_id` 指向远端 P 节点。
- `src_list` 是本地 D 侧目标地址列表。
- `dst_list` 是远端 P 侧源地址列表。
- `batch_transfer_sync_read` 表示“从远端 dst 读到本地 src”。

Mamba group 会调用 `_append_mamba_transfer_meta()`，按 conv / ssm 两类 state 分别计算地址和长度，并处理 TP ratio 切片。

## 5.9 执行 Mooncake 传输

```python
ret = self.engine.batch_transfer_sync_read(
    session_id,
    src_list,
    dst_list,
    length_list,
)
if ret < 0:
    raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")
```

成功后，D 侧 KV Cache 已经写入本地 NPU KV buffer。

## 5.10 传输后的 reformat

如果 D 侧从多个 P TP shard 拼出自己的 KV head，或需要 NZ layout，会在最后一个 group shard 完成时做 reformat：

```python
if need_cat_cache:
    self.reformat_kv_cache_with_fused_op(...)
if need_nz_cache:
    self.reformat_kv_cache(...)
```

场景：

- `num_group_pulls > 1`：GQA/MHA 非对称 TP，需要按 head 重新拼接。
- `enable_kv_nz=True`：需要转换为 NPU NZ 期望布局。
- HMA/hybrid-linear 场景会调用 `reformat_kv_cache_hybrid_linear_torch()`。

## 5.11 通知 P 侧完成

每个 shard 最后都会向 P 侧发 `DONE_RECVING_MSG`：

```python
data_bytes = self.encoder.encode((DONE_RECVING_MSG, request_id, remote_port_send_num))
ensure_zmq_send(sock, data_bytes, f"{remote_host}:{remote_handshake_port}")
resp = ensure_zmq_recv(sock, self.remote_poller, path)
if resp != b"ACK":
    raise RuntimeError(...)
```

P 侧收到后，会更新 task tracker；当所有需要的 D 侧拉取都完成后，P 侧延迟 blocks 可以释放。

## 5.12 失败处理

接收线程维护：

```python
self.failed_recv_requests: set[str]
self.invalid_block_ids: set[int]
```

失败时：

```python
def _mark_failed_recv_request(self, request_id, local_block_ids):
    self.failed_recv_requests.add(request_id)
    self.invalid_block_ids.update(local_block_ids[0])
```

顶层连接器暴露：

```python
def get_block_ids_with_load_errors(self):
    return self.kv_recv_thread.get_and_clear_invalid_block_ids()
```

这让 Scheduler 能识别加载失败的本地 blocks，避免错误复用。
