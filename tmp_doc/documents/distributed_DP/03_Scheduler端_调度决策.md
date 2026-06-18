# 第 3 章：Scheduler 端 - 调度决策

P/D 分离的 Scheduler 端围绕两个方向展开：

1. **P 侧**：请求完成 prefill 后，返回 D 侧拉取 KV 所需的 `kv_transfer_params`，并决定是否延迟释放 blocks。
2. **D 侧**：收到带 `do_remote_prefill=True` 的请求后，把远端 KV 视作“外部已计算 token”，预分配本地 blocks，并构建 Worker 侧拉取任务。

核心类：`code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` 中的 `MooncakeConnectorScheduler`。

## 3.1 初始化

```python
class MooncakeConnectorScheduler:
    def __init__(self, vllm_config, engine_id, kv_cache_config):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        init_ascend_config(vllm_config)
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        self.local_ip = get_ip()

        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + data_parallel_rank * tensor_parallel_size * pipeline_parallel_size * self.pcp_size
        )

        self._reqs_need_recv = {}
        self._reqs_need_send = {}
        self._reqs_in_batch = set()
        self.multi_nodes_meta_mapping = {}
```

### 3.1.1 状态表

| 字段 | 作用 |
|------|------|
| `_reqs_need_recv` | D 侧：记录需要 Worker 拉远端 KV 的请求 |
| `_reqs_need_send` | P 侧：记录已经返回给 D、需要延迟释放的请求 |
| `_reqs_in_batch` | 当前 step 中涉及 remote prefill/decode 的请求，Worker 用它初始化 task tracker |
| `multi_nodes_meta_mapping` | 跨节点 rank → host/engine_id 映射，由 handshake metadata 填充 |
| `group_transfer_info` | 每个 KV group 的 block 粒度、SWA window、是否 state group 等信息 |

## 3.2 D 侧：get_num_new_matched_tokens

D 节点收到 Proxy 转发的请求时，`kv_transfer_params` 中有 `do_remote_prefill=True`。

```python
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    params = request.kv_transfer_params

    if params is not None and params.get("do_remote_prefill"):
        token_ids = request.prompt_token_ids or []
        actual = self._state_prefill_token_count(len(token_ids))
        params["num_computed_tokens"] = num_computed_tokens
        count = max(actual - num_computed_tokens, 0)
        if count > 0:
            return count, True

    if params is not None and params.get("do_remote_decode") and self.need_truncate:
        self._truncate_request_for_prefill(request)

    return 0, False
```

关键点：

- `do_remote_prefill=True` 表示 D 侧要加载远端 KV。
- 返回 `(count, True)`，其中 `True` 表示异步加载 remote KV；Scheduler 会把请求置为 `WAITING_FOR_REMOTE_KVS`。
- `count` 是 D 侧需要为 remote KV 预留的 token 数。
- 对 Mamba / compressed / state group 场景，`_state_prefill_token_count()` 可能返回 `N-1`，因为 D 侧会重算最后一个 token。

## 3.3 P 侧：请求截断逻辑

当请求发给 P 节点，`kv_transfer_params` 中有 `do_remote_decode=True`。如果模型需要 state 截断（如 Mamba/hybrid/compress 场景），P 侧会把 prompt 最后一个 token 去掉：

```python
def _truncate_request_for_prefill(self, request):
    params = request.kv_transfer_params
    if params and not params.get("_p_side_truncated") and request.num_prompt_tokens > 1:
        request.prompt_token_ids.pop()
        request._all_token_ids.pop()
        request.num_prompt_tokens -= 1
        request.max_tokens = 1
        params["_p_side_truncated"] = True
```

原因：D 侧需要从 `h(N-1)` 开始，重算最后一个 token 得到正确的 `h(N)`。这样可以避免 state group（例如 Mamba）在 P/D 边界上语义错位。

## 3.4 D 侧：update_state_after_alloc

Scheduler 分配本地 KV blocks 后，调用：

```python
def update_state_after_alloc(self, request, blocks, num_external_tokens):
    params = request.kv_transfer_params

    if params is not None and (params.get("do_remote_prefill") or params.get("do_remote_decode")):
        self._reqs_in_batch.add(request.request_id)

    if params is not None and params.get("do_remote_prefill"):
        if params.get("remote_block_ids"):
            if all(p in params for p in ("remote_engine_id", "remote_host", "remote_port", "remote_request_id")):
                local_block_ids = blocks.get_unhashed_block_ids_all_groups() if num_external_tokens > 0 else []
                self._reqs_need_recv[request.request_id] = (
                    request,
                    local_block_ids,
                    num_external_tokens,
                )
        else:
            assert num_external_tokens == 0

        params["do_remote_prefill"] = False
```

含义：

- `local_block_ids` 是 D 侧新分配、准备写入远端 KV 的本地 blocks。
- `remote_block_ids` 来自 P 侧 `request_finished()` 返回的 `kv_transfer_params`。
- `do_remote_prefill` 会被置为 `False`，确保同一个请求只触发一次 remote pull。

## 3.5 build_connector_meta

Scheduler 每个 step 构建传给 Worker 的 metadata：

```python
def build_connector_meta(self, scheduler_output):
    meta = MooncakeConnectorMetadata()

    for req_id, (req, block_ids, num_external_tokens) in self._reqs_need_recv.items():
        meta.add_new_req(
            request_id=req_id,
            local_block_ids=block_ids,
            num_external_tokens=num_external_tokens,
            kv_transfer_params=req.kv_transfer_params,
        )

    self._reqs_need_recv.clear()
    meta.requests_to_send = self._reqs_need_send
    self._reqs_need_send = {}
    meta.reqs_in_batch = self._reqs_in_batch
    self._reqs_in_batch = set()

    return meta
```

`MooncakeConnectorMetadata` 包含：

```python
class MooncakeConnectorMetadata(KVConnectorMetadata):
    requests: dict[str, ReqMeta]
    requests_to_send: dict[str, float]
    reqs_in_batch: set[str]
```

其中 `ReqMeta` 会保存：

| 字段 | 说明 |
|------|------|
| `local_block_ids` | D 侧本地要写入的 KV blocks |
| `num_external_tokens` | 需要从 P 侧拉取的 token 数 |
| `num_computed_tokens` | D 侧已本地计算 token 数，用于 prefix cache 差异修正 |
| `remote_block_ids` | P 侧提供的远端 KV blocks |
| `remote_engine_id` | P 侧 engine id |
| `remote_request_id` | P 侧请求 id |
| `remote_host` / `remote_port` | P 侧 side-channel 地址 |
| `remote_pcp_size` / `remote_dcp_size` / `remote_ptp_size` | P 侧并行信息 |
| `remote_multi_nodes_meta_mapping` | 跨节点 rank metadata |
| `num_prompt_blocks` | P 侧 prompt block 数，用于 CP/prefix-cache 映射 |

## 3.6 P 侧：request_finished

P 节点 prefill 请求完成后，Scheduler 调用 `request_finished()`。

```python
def request_finished(self, request, block_ids):
    params = request.kv_transfer_params

    if (
        params is None
        or not params.get("do_remote_decode")
        or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
    ):
        return False, None

    num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)
    computed_block_ids = self._get_transfer_block_ids(block_ids, len(request.prompt_token_ids))
    computed_block_ids = self._get_swa_transfer_block_ids(computed_block_ids)
    delay_free_blocks = sum(len(x) for x in computed_block_ids) > 0

    if delay_free_blocks:
        self._reqs_need_send[request.request_id] = time.time()

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

关键点：

- 只有 `do_remote_decode=True` 且请求以 `FINISHED_LENGTH_CAPPED` 结束时，才返回远端 KV 信息。
- `delay_free_blocks=True` 表示 P 侧 blocks 暂时不能释放，要等 D 侧拉取完成后再释放。
- 返回的 dict 会成为发送给 D 节点请求中的 `kv_transfer_params`。

## 3.7 block 裁剪

### 3.7.1 `_get_transfer_block_ids`

用于去掉不属于 prompt KV 的 blocks，例如 speculative/MTP 额外 blocks。

- Attention-like group：根据 `prompt_len / tokens_per_block` 截取。
- Mamba/state group：不是普通 context block 对齐，直接保留。

### 3.7.2 `_get_swa_transfer_block_ids`

SlidingWindow group 只保留 window tail，并去掉 placeholder block 0。

```python
window_blocks = blocks[-group_info.blocks_per_window:]
transfer_block_ids.append([block_id for block_id in window_blocks if block_id != 0])
```

## 3.8 调度状态 WAITING_FOR_REMOTE_KVS

以 `scheduler_dynamic_batch.py` 为例：

```python
if load_kv_async:
    skipped_waiting_requests.prepend_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    continue
```

后续调度循环会检查：

```python
if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
    is_ready = self._update_waiting_for_remote_kv(request)
    if is_ready:
        request.status = RequestStatus.WAITING
    else:
        skipped_waiting_requests.prepend_request(request)
        continue
```

因此 D 侧请求在 remote KV 未拉取完成前不会进入正常 RUNNING；当 Worker 通过 `get_finished()` 上报 done_recving 后，它才回到可调度状态。
