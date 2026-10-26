# 04. Worker Pipeline：从 Metadata 到设备任务

基线与源码链接见 [00](../00_source_baseline.md)。主要阅读 `pool_worker.py`，再进入 `kv_transfer.py`。

## 1. Worker 连接四种坐标

```text
Request/token/block ids
  + layer name/group/cache entries
  + local base/stride/NPU event
  + external key/GVA/range/session
  -> 实际 load/save
```

Scheduler 决定请求和 block allocation，Worker 持有真实 tensor。Worker 不重新分配 Scheduler 的 KV blocks，但要管理 backend session、lease、partial snapshot 和传输队列。

## 2. 初始化先决定数据面

```text
use_layerwise=False
  -> 普通 object bulk

use_layerwise=True:
  layerwise_data_plane=="gva"       -> MemCache GVA
  layerwise_data_plane=="block_key" -> Mooncake session/range
  无上述能力                       -> KeyLayer object 分支
```

`use_layerwise_transfer` 当前专指 GVA；`use_block_key_layerwise` 专指 block-key。不能只检查前者就断言“没有 layerwise 数据面”。

Worker 初始化还推导：
- TP/PP/PCP/DCP/KVPP 坐标、put leader、head 切分。
- cacheable group、family、group page/hash/transfer 粒度。
- Mamba align/recurrent/private state。
- load/save 角色及 decode-only 保存能力。
- layerwise protocol、topology/runtime 校验、physical layout。

Scheduler 和 Worker 的逻辑身份必须一致；实际 tensor 地址只在 Worker 绑定。

## 3. register_kv_caches：初始化数据面

### 3.1 从真实 tensor 推导布局

`_get_cache_block_metadata()` 区分 tensor block 数、logical page scale、有效 block bytes、stride 和 region length。多个 K/V/state/scale/indexer tensor 可属于同一 physical layer。

`_infer_cache_group_metadata()` 建立：

```text
group_kv_caches_base_addr
group_block_len / group_block_stride
group_layer_cache_entry_offsets
group_num_layers
physical_layer_to_group_layers
```

这不是“一层一个 pointer”的结构；一层可能有多个 entries。

### 3.2 本地内存注册

`_get_storage_key()` 用底层 storage pointer 去重；同 storage views 的范围被合并后注册。hybrid 的 `_align_kv_ptrs()` 处理 backend 要求的 2 MB 对齐并校验 storage 边界。

```text
tensor storage/layout 推导
  -> ChunkedTokenDatabase.set_group_buffers()
  -> backend.ensure_initialized（适用时）
  -> backend.register_buffer(ptrs, lengths)
  -> backend 能力校验 / layerwise builder
  -> 启动并等待 transfer thread ready
```

注册的是本地 HBM 访问区，backend 对象容量另行分配。多个 layer alias 相同 storage 不等于它们可随意共享 KV 内容。

## 4. 实际创建哪些线程

| 模式 | 保存 | 加载 |
|---|---|---|
| bulk object | `KVCacheStoreSendingThread` | async 时 `KVCacheStoreRecvingThread`；同步直接 get |
| block-key layerwise | `KVCacheStoreLayerSendingThread` | `KVCacheStoreLayerRecvingThread` |
| GVA layerwise | 同上 | 同上 |
| KeyLayer object | `KVCacheStoreKeyLayerSendingThread` | `KVCacheStoreKeyLayerRecvingThread` |

发送线程受角色门控；load-only consumer 不应等待不存在的 send thread。六个派生类仍存在，但 GVA 和 block-key 共用 Layer 线程，由任务类型走不同执行分支。

## 5. 每轮入口：先 fence，再准备，再使用

runner 在新一轮前调用 `handle_preemptions()`，Store 按 `released_req_ids` 等待即将被覆盖的旧 bulk 保存。

layerwise 的 `bind_connector_metadata()` 提前调用 `prepare_layerwise_step()`：
1. drain 上轮可能延后的 final save。
2. 对 GVA reuse 的 replicated TP 场景按配置条件 barrier。
3. 重置 `current_layer`、预取 cursor、attention gate、失败门控。
4. `process_layer_data()` 建立当前 step tasks。

layerwise `start_load_kv()` 随后直接返回，避免后置启动点重置已推进的 target/MTP layer cursor。bulk `start_load_kv()` 则实际准备或提交 get。

## 6. Bulk load/save

### 6.1 Load

每个可 load 的 ReqMeta：
- 依据外部命中与最后 token 规则确定实际读取长度。
- 从本地 cached watermark 之后生成 key/block/address/size。
- 应用 group load mask，跳过 Mamba null block。
- 同步模式直接 `Backend.get`；async 模式把请求交给 recv thread。
- TP mismatch 进入专门的 head-slice/strided address 路径。
- 循环移位 key/address/size/block id 时维持逐项配对。

MemCache 失败可记录 request-level failed；其他返回码也可能映射为 invalid block ids。hybrid ids 不能不加区分地塞进旧单 group 回退协议。

### 6.2 Save

bulk `wait_for_save()`：
```text
筛选 can_save
  -> 一批请求共用当前 NPU event
  -> add_save_batch
  -> 后台等待 producer event、生成增量 key/address、put
```

方法名不表示立即 join。发送线程用成功保存 offset、stored request 计数、重试 token 信息，处理同 request 多 chunk、部分失败和后续重试。

覆盖保护发生在后续 released fence。不要把“本轮 batch 已提交”写成“本轮所有 KV 已持久化”。

## 7. Layerwise 任务准备顺序

`process_layer_data()` 每轮建立新的 load/save task 列表，避免上轮线程迟到的 clear 删除新任务。

```text
计算 reachable store masks
  -> block-key: prepare_layerwise_sessions
  -> 各 physical layer / group 构建 save ranges
  -> 先保护/取得旧 load GVA
  -> 再分配新 save GVA
  -> build_shared_save_data
  -> 构建 load ranges
  -> build_shared_load_data
```

先处理旧 partial load，再分配新 snapshot，防止后者覆盖恢复仍依赖的旧尾块。

同一 physical layer 可映射多个 `(group_id, layer_idx_in_group)`。mask 会把连续区间拆成多个有效 runs，partial block 只附加一次。

## 8. Layerwise load 起点按模式解释

### 8.1 MemCache GVA

无物理 reuse，或当前层独立：通常从本地 cached block 后开始。

共享 buffer 的层：必须从 block 0 恢复该层历史 KV，因为本地 hash 账本命中不证明当前 slot 还保存这层内容。partial 与 rollback snapshot 走专门的 GVA/offset 逻辑。

### 8.2 Mooncake block-key

从 `load_block_keys` 中已填充 slot 产生 range。`LayerwiseSessionTracker` 保留前面 chunk 已 commit 的 prefix，所以后续 chunk 即使没有新的 LoadSpec，仍可能需要恢复之前的 committed objects。

不能把 GVA 的一个 `load_start_block` 公式推广到 block-key 全部路径。

## 9. 每层与 attention 交错

`_submit_ready_layer_loads()` 依据预取窗口发任务，包含上一 slot owner、attention gate、external slot release 等依赖。

`wait_for_layer_load()`：
1. 检查线程异常。
2. 为 block-key hybrid 设置 attention on_start/on_finish callbacks；recurrent 层保持计算后保存。
3. 提交当前 demand load 与后续预取。
4. 等本层 load event，周期性检查失败。
5. 检查 hybrid 恢复完整性；必要时 abort、取消 gate、drain、关闭 session。
6. 完成后清本层 event，处理最终 load session。

connector 随后执行 Mamba state copy。V1 传 copy buffers，V2 可传 mamba model state；copy 必须在对应 load 之后。

`save_kv_layer()`：
- 尚未由 attention callback 提交时，记录 producer event 并提交当前层。
- recurrent/record-only 路径运行 transfer window 收尾。
- model final layer 默认等待队列与保存 event；空末层也不能漏掉前层 I/O。
- 最后推进 cursor。

## 10. 完成、失败和 teardown

`get_finished()` 对 bulk async load 收集接收完成；普通发送集合被内部消费，向 Scheduler 返回的 `done_sending` 为空。block-key terminal 请求释放 session owner；GVA step load keys 清 lease。

`get_transfer_results()` 额外取 failed request；bulk async 的失败也要结束接收等待，防止请求永远挂在 waiting-remote。

错误分成：
- 单 block invalid：交给对应上游回退策略。
- request-level `failed_recving`：按整个请求恢复失败处理。
- fatal transfer exception：`raise_if_failed()` 在同步点抛出。

shutdown 先停止使用者、结束线程/lookup/session，再关闭 native backend 资源。实现细节与资源顺序见 [05](05_transfer_backend_storage.md)、[06](06_concurrency_and_config.md)。
