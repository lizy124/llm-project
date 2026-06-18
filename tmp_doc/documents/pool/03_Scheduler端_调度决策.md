# 第 3 章：Scheduler 端 - 调度决策

Scheduler 端负责回答三个核心问题：
1. **这个请求有没有命中外部池中的 KV Cache？**（查找）
2. **命中后，需要预留多少本地 block？**（分配）
3. **哪些请求的 KV Cache 需要存到外部池？**（构建元数据）

## 3.1 KVPoolScheduler 初始化

[`KVPoolScheduler`](../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py) 在构造时完成以下关键初始化：

```python
class KVPoolScheduler:
    def __init__(self, vllm_config, use_layerwise, kv_cache_config=None):
        # 1. 读取模型配置
        self.use_hybrid = self._uses_hybrid_kv_cache(...)  # 是否混合 KV Cache 组
        self.compress_ratios = ...  # 压缩比率（如 DeepSeek V4 的 c4/c128）
        
        # 2. 计算 block 大小
        self.original_block_size = self._infer_group_block_sizes(...)  # 原始 block_size
        cp_scale = self.pcp_size * self.dcp_size  # Context Parallel 缩放
        self.grouped_block_size = [bs * cp_scale for bs in self.original_block_size]
        self.hash_block_size = ...  # hash 用的 block 大小
        self.lcm_block_size = math.lcm(*self.grouped_block_size)  # 最小公倍数
        
        # 3. 计算传输粒度
        self.cache_transfer_granularity = self._infer_cache_transfer_granularity()
        # 例如：DeepSeek V4 压缩组 c1=128, c4=512, c128=16384
        # 最终 granularity = lcm(128, 512, 16384) = 16384
        
        # 4. 创建 LookupKeyClient（ZMQ 客户端，用于查询 Worker 端的池命中情况）
        self.client = LookupKeyClient(vllm_config)
        
        # 5. 状态追踪
        self.load_specs: dict[str, LoadSpec] = {}       # 请求 → 加载规格
        self._request_trackers: dict[str, RequestTracker] = {}  # 请求追踪器
        self._unfinished_requests: dict[str, ...] = {}  # 未完成的请求
```

### 3.1.1 cache_transfer_granularity 的计算

```python
def _infer_cache_transfer_granularity(self) -> int:
    granularities = [self.lcm_block_size]
    for group_id in self.kv_cache_group_ids:
        granularities.append(
            get_cache_family_granularity(
                self._get_group_block_size(group_id),
                self._get_group_family(self.kv_cache_group_families, group_id),
            )
        )
    return math.lcm(*granularities)
```

对于 DeepSeek V4（有 c1/c4/c128 三个压缩组），`get_cache_family_granularity` 计算如下：
- c1 × block_size(128) = 128
- c4 × block_size(128) = 512
- c128 × block_size(128) = 16384
- 最终 `cache_transfer_granularity = lcm(128, 128, 512, 16384) = 16384`

这意味着 KV Cache 的存取必须以 16384 tokens 为最小单位对齐。

## 3.2 命中检测：get_num_new_matched_tokens

这是 Scheduler 调度新请求时调用的第一个关键方法。

```python
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    # 1. 如果当前角色是 consumer 且不需要 load，直接返回 0
    if self.kv_role == "kv_consumer" and not self.consumer_is_to_load:
        return 0, False
    
    # 2. 对齐到 cache_transfer_granularity
    if self._discard_partial_chunks:
        token_len = self._floor_to_cache_transfer_granularity(
            len(request.prompt_token_ids)
        )
    
    # 3. 如果 token 太短，跳过
    if token_len < self.cache_transfer_granularity:
        return 0, False
    
    # 4. ★ 通过 ZMQ 调用 Worker 端的 lookup_scheduler()
    num_external_hit_tokens = self.client.lookup(
        token_len,
        request.block_hashes,
        self.kv_cache_group_ids,
    )
    
    # 5. 如果完全命中（等于总 token 数），减少一个 token（避免 decode 阶段无输入）
    if num_external_hit_tokens == request.num_tokens:
        num_external_hit_tokens -= 1
    
    # 6. 计算实际需要加载的 token 数
    if num_external_hit_tokens < num_computed_tokens:
        need_to_allocate = 0
    else:
        need_to_allocate = num_external_hit_tokens - num_computed_tokens
    
    # 7. 没有新增外部 token 需要加载时，不创建 LoadSpec
    if need_to_allocate <= 0:
        return 0, False
    
    # 8. 创建 LoadSpec（记录外部命中和本地已计算的 token 数）
    self.load_specs[request.request_id] = LoadSpec(
        vllm_cached_tokens=num_computed_tokens,
        kvpool_cached_tokens=num_external_hit_tokens,
        can_load=False,  # 暂时还不能加载，需要等分配 block 后
    )
    
    return need_to_allocate, self.load_async and not self.use_layerwise
```

### 3.2.1 LookupKeyClient 的 ZMQ 通信

```python
class LookupKeyClient:
    def lookup(self, token_len, block_hashes, kv_cache_group_ids):
        # 1. 将 block_hashes 编码为 hex 字符串
        hash_strs = [h.hex() for h in block_hashes]
        
        # 2. 通过 ZMQ REQ socket 发送查询
        all_frames = [token_len_bytes] + list(kv_group_frames) + list(hash_frames)
        self.socket.send_multipart(all_frames, copy=False)
        
        # 3. 接收响应（hit 的 token 数量）
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result
```

Scheduler 端通过 ZMQ IPC 与 Worker 端的 `LookupKeyServer` 通信，后者调用 `KVPoolWorker.lookup_scheduler()` 实际查询外部池。

## 3.3 分配后状态更新：update_state_after_alloc

当 Scheduler 为请求分配了 KV Cache block 后，调用此方法更新状态。

```python
def update_state_after_alloc(self, request, blocks, num_external_tokens):
    # 1. 获取本地 block IDs
    local_block_ids = normalize_block_ids_by_group(blocks.get_block_ids())
    
    # 2. 记录未完成的请求
    self._unfinished_requests[request.request_id] = (request, local_block_ids)
    
    # 3. 如果没有外部命中，直接返回
    if request.request_id not in self.load_specs:
        return
    
    # 4. 如果 num_external_tokens == 0，禁用加载
    if num_external_tokens == 0:
        self.load_specs[request.request_id].can_load = False
        return
    
    # 5. 验证 token 数量一致
    assert num_external_tokens == (
        self.load_specs[request.request_id].kvpool_cached_tokens
        - self.load_specs[request.request_id].vllm_cached_tokens
    )
    
    # 6. ★ 启用加载标志
    self.load_specs[request.request_id].can_load = True
```

## 3.4 构建元数据：build_connector_meta

每次调度步骤中，Scheduler 调用此方法构建传递给 Worker 的元数据。

```python
def build_connector_meta(self, scheduler_output):
    # 1. 清理已完成的请求
    for finished_req_id in scheduler_output.finished_req_ids:
        self._request_trackers.pop(finished_req_id, None)
        self._unfinished_requests.pop(finished_req_id, None)
    
    # 2. 创建 AscendConnectorMetadata
    meta = AscendConnectorMetadata(unfinished_request_ids, preempted_req_ids)
    
    # 3. 处理新调度的请求
    for request in scheduler_output.scheduled_new_reqs:
        load_spec = self.load_specs.pop(request.req_id, None)
        
        # 创建 RequestTracker
        request_tracker = RequestTracker(
            req_id=request.req_id,
            token_len=num_tokens_to_compute,
            allocated_block_ids_by_group=...,
            token_ids=request.prompt_token_ids[:num_tokens_to_compute].copy(),
        )
        
        # 从 RequestTracker 构建 ReqMeta
        req_meta = ReqMeta.from_request_tracker(
            request_tracker,
            self.cache_transfer_granularity,
            load_spec=load_spec,
            skip_save=force_skip_save,
            block_hashes=request_real.block_hashes,
            is_last_chunk=...,
            discard_partial_chunks=self._discard_partial_chunks,
            original_block_size=self.original_block_size,
            kv_cache_group_families=self.kv_cache_group_families,
        )
        meta.add_request(req_meta)
    
    # 4. 处理缓存的请求（chunked prefill 的后续 chunk）
    # 类似逻辑，更新 RequestTracker 并创建新的 ReqMeta
    
    return meta
```

### 3.4.1 ReqMeta.from_request_tracker 的关键逻辑

```python
@staticmethod
def from_request_tracker(tracker, cache_transfer_granularity, load_spec, skip_save, ...):
    input_token_len = tracker.token_len
    
    # 决定是否保存
    chunk_boundary = (
        cdiv(tracker.num_saved_tokens + 1, cache_transfer_granularity)
        * cache_transfer_granularity
    )
    num_tokens_to_save = (
        input_token_len // cache_transfer_granularity * cache_transfer_granularity
    )
    
    skip_save = skip_save or num_tokens_to_save < chunk_boundary
    
    if not skip_save:
        tracker.num_saved_tokens = num_tokens_to_save  # 更新已保存 token 数
    
    return ReqMeta(
        req_id=tracker.req_id,
        token_len_chunk=num_tokens_to_save,
        block_ids_by_group=tracker.allocated_block_ids_by_group,
        can_save=not skip_save,
        load_spec=load_spec,
        block_hashes=block_hashes,
        is_last_chunk=is_last_chunk,
        token_ids=token_ids,
        original_block_size=original_block_size,
        kv_cache_group_ids=list(range(len(tracker.allocated_block_ids_by_group))),
        kv_cache_families_by_group=kv_cache_group_families,
    )
```

## 3.5 Scheduler 端完整调用链

```
vLLM Scheduler 调度循环
        │
        ▼
get_num_new_matched_tokens(request, num_computed_tokens)
        │
        │  ZMQ REQ → LookupKeyServer (Worker)
        │  ← 返回 hit 的 token 数量
        │
        ▼
Scheduler 根据 hit 数量决定分配多少 block
        │
        ▼
update_state_after_alloc(request, blocks, num_external_tokens)
        │  → 设置 load_specs[req_id].can_load = True
        │
        ▼
build_connector_meta(scheduler_output)
        │  → 构建 AscendConnectorMetadata（包含 ReqMeta 列表）
        │  → 传递给 Worker 端
        │
        ▼
Worker 端 start_load_kv() / wait_for_save()
```