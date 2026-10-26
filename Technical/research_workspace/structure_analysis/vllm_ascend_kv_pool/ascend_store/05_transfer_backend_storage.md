# 05. 存储模型、传输线程与 Backend 协议

基线与源码入口见 [00](../00_source_baseline.md)。主要文件：`kv_transfer.py`、`backend/base.py`、各 backend 和 protocol 模块。

## 1. 两侧内存与两种外部寻址

本地侧是 vLLM 分配的 NPU KV tensor；`register_buffer()` 让 backend 认识这些可读写区间。外部侧由 SDK/服务管理，可能是 host DRAM，也可能启用 SSD 层级。不能把本地 HBM 注册容量当作外部池容量。

| 描述 | 本地侧 | 外部侧 |
|---|---|---|
| bulk object | 每 key 对应多个地址片段 | key 定位完整 value |
| MemCache GVA | 当前 layer 的 HBM 地址 | GVA + layer/entry offset |
| Mooncake range | 当前 layer 的 HBM 地址 | key + 对象内 range offsets |
| partial snapshot | 未完整 token page 的本地内容 | 独立身份/临时对象，不能冒充完整 hash 对象 |

MemCache 的 GVA 对象通常包含一组相关层的数据；逐层传输通过对象内偏移定位。GVA 与“每层单独对象”不是同义词。

## 2. 能力由注册表与 backend 显式声明

`backend_map` 当前注册 Mooncake、MemCache、YuanRong。`layerwise_protocol` 保存模块路径，`get_layerwise_data_plane()` 读取模块的 `LAYERWISE_DATA_PLANE`：

| backend | layerwise plane | 普通 get/put | 专用能力 |
|---|---|---|---|
| Mooncake | `block_key` | 有 | put/get session、range transfer、commit/revoke |
| MemCache | `gva` | 有 | key info、alloc、lease、write finish、GVA copy |
| YuanRong | 无专用 plane | 有 | object API；走保留的 KeyLayer 分支时不具备上述专用能力 |

基类对不支持的扩展抛 `NotImplementedError`。不要因为所有对象都叫 Backend，就假定它们有相同发布、错误码和内存注册语义。

Scheduler client 是独立实例：Mooncake 不贡献池内存，MemCache 不初始化 Worker 的存储资源。两者可以查元数据，但不持有 Worker 注册的 KV tensor 地址。

## 3. 最小对象协议

```text
exists / batch_is_exist(keys)
put(keys, [[addr...], ...], [[size...], ...])
get(keys, [[addr...], ...], [[size...], ...])
```

二维列表保持：
```text
key[i] <-> addrs[i] <-> sizes[i]
```
一 key 的 value 可以来自多层、K/V 两个 tensor、state/scale 等多个不连续区域。循环移位、去重、mask 过滤都必须保持三者配对。

`put()` 的统一表面允许 bool、per-key bool list，以及遗留的 None（通过异常报告失败）；get 和专用批量协议有自己的返回约定。不能对所有方法都用 `all(result == 0)`。

`requires_exists_before_put` 表示 connector 是否需要在写入前过滤已存在 key；应服从 backend 声明，不能为所有后端硬加一轮 exists。

## 4. Mooncake block-key：session 与 range

当前 Mooncake layerwise 先调用 `validate_layerwise_support()` 确认客户端有 session/range APIs。旧 SDK 即使支持普通 get/put，也可能不支持此模式。

### 4.1 保存

```text
Worker 为每 block key 决定 object_size
  -> batch_put_start
  -> 每层 LayerRangeReqMeta
  -> batch_copy_put(keys, buffers, sizes, dst_offsets)
  -> 该 group 的最后实际层完成
  -> batch_commit
  -> session tracker 将 key 转入 owner 的未来 load 集合
```

`batch_commit` 映射 SDK 的 put session end。开放写会话中的对象不能仅因已分配就作为完整可读对象命中。

range copy 失败或 commit 失败时 revoke 相关 key，防止部分写入对象留下可误用身份。异常清理还要覆盖本 forward 中已经 start、但尚未完成的其他对象。

### 4.2 加载

```text
batch_is_readable 检查 committed key
  -> batch_get_start
  -> 每层 batch_copy_get(keys, buffers, sizes, src_offsets)
  -> 相关 range 调用完成
  -> 最后相关 chunk/owner 退出时 batch_get_end
```

`LayerwiseSessionTracker` 处理多个 request 共享同 key、chunked prefill 后续恢复、失败重试和 terminal owner 清理。get session 可跨 chunk 保留，不是每个调度 step 结束都无条件关闭；不能让某个请求的重试关闭另一个请求仍使用的 get session。

### 4.3 返回形状

`require_aligned_batch_results()` 检查一 key 一结果且为整数，拒绝 None、数量不符、布尔值冒充整数等异常。各操作成功条件不同：copy 分支检查负值，commit 检查非零。读 SDK wrapper 和调用方时必须一起核对，不能只看类型。

## 5. MemCache GVA：alloc、lease 与发布

### 5.1 保存

```text
完整/partial key
  -> batch_alloc(keys, object_sizes, ttl)
  -> GVA + 每层 offset
  -> NPU producer event 完成
  -> COPY_L2G: HBM -> 外部地址
  -> 所有相关层 copy 完成
  -> batch_write_finish(keys, results)
```

“分配到 GVA”“copy 完成”“发布完成”是三个阶段。写入失败必须按失败协议清理/撤销可误用的对象状态，不能发布部分成功为完整数据。

### 5.2 加载

```text
batch_get_key_info(for_load=True) / lease
  -> 校验可用 GVA
  -> COPY_G2L: 外部地址 -> 本层 HBM
  -> layer event
  -> 全部相关 copy 完成后移除本轮 load lease
```

Scheduler 查询使用 `batch_is_readable()`；当前实现要求正 size、有效非零 GVA。仅有 size 并不足以承诺直接恢复。

### 5.3 SSD 的边界

`batch_alloc()` 可能返回 SSD hit sentinel，不能当普通 GVA 发起 DMA。`batch_get_key_info(for_load=True)` 对 SSD-only 且没有有效 GVA 的对象，使用注册的临时 NPU buffer 走普通 Get，等待 rewarm 后重新查询。

**Scheduler 的 readability 查询本身不会调用这个 for-load rewarm。** 因此某个 SSD-only 对象可能在 lookup 阶段保守地被判不可读；不要把“SDK 能恢复 SSD”写成“所有 SSD 对象都会被这条 lookup 命中”。

临时 buffer 的创建、注册、Get、注销属于恢复路径成本，也不能混入纯元数据查找的性能理解。

## 6. LayerBatchBuilder 的两种输出

`build_shared()` 先处理同组跨层不变的 block rows、keys/GVA、request ids，再由 `build_addrs()` 加本层本地地址。

| 输出 | 组织方式 | copy 输入 |
|---|---|---|
| `LayerBatchReqMeta` | 扁平 NumPy arrays | GVA/local address/size |
| `LayerRangeReqMeta` | 每 key 的多个 buffer/range | key/buffers/sizes/offsets |

Builder 复用内部数组并做去重；同一保存对象不能因为多个共享 prefix/request range 而重复发布或错误扣计数。去重后的 request 完成与对象完成仍要独立记录。

batch 限制按 `layerwise_max_transfer_blocks` / bytes 拆包。拆包不能提前触发 commit/write finish：完整对象需要的所有相关包和层都成功才可发布。

## 7. 六类传输线程

`KVTransferThread` 提供设备设置、队列、提交锁、停止标记、异常槽、finished/failed 集合、event 与 batch helpers。

| 派生类 | 输入 | 主要完成边界 |
|---|---|---|
| `KVCacheStoreSendingThread` | ReqMeta / batch fence | bulk request 保存 job，成功 offset 和 Mamba completed event |
| `KVCacheStoreRecvingThread` | ReqMeta | bulk async request 接收完成 |
| `KVCacheStoreKeyLayerSendingThread` | LayerTransferTask 列表 | object 逐层 put 与 layer event |
| `KVCacheStoreKeyLayerRecvingThread` | LayerLoadTask | object 逐层 get 与 layer event |
| `KVCacheStoreLayerSendingThread` | GVA 或 range tasks | GVA write finish / block-key commit 与 layer event |
| `KVCacheStoreLayerRecvingThread` | GVA 或 range load tasks | load layer event、invalid/failed 记录 |

普通发送线程还有 `stored_requests` 的 job 账本与 `wait_for_requests_saved()`，用于 released block fence。layerwise 发送 base 则按 layer/group task 维护引用计数；不要混用两者的计数语义。

## 8. 失败不是一类结果

| 情况 | 当前处理边界 |
|---|---|
| 外部没有对象 | lookup miss，按正常计算路径推进 |
| 返回长度/类型不对 | batch shape error 或相应校验 |
| 普通 get 某 key 失败 | invalid blocks 或 request failed，依路径而定 |
| MemCache multi-group 恢复失败 | request-level failed；禁用当前保存，避免发布坏 KV |
| block-key hybrid 恢复不完整 | abort/fail，拒绝将不完整状态送入 attention |
| 保存失败 | 不推进成功水位/不发布；重试或 revoke/失败记录 |
| 后台 fatal exception | 记录异常，等待/轮询处 `raise_if_failed` |
| 等待 event 周期性超时 | 检查错误、打印后继续等待，不视为成功 |

`stop()` 与提交锁防止关闭/失败后继续挂入无人消费的任务；`shutdown()` 必须先停止 store 使用者，再释放 native backend。并发顺序见 [06](06_concurrency_and_config.md)。
