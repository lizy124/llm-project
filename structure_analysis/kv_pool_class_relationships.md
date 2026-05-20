# KV Pool 类关系图

## 一、继承关系

### 1. 连接器基类继承

```
KVConnectorBase_V1 (vLLM基类)
    ├── AscendStoreConnector
    │   └── 管理分布式KV缓存存储
    │
    ├── CPUOffloadingConnector
    │   └── 管理GPU↔CPU KV缓存交换
    │
    ├── LMCacheConnectorV1
    │   └── 集成LMCache库
    │
    └── UCMConnectorV1
        └── UCM连接器实现
```

### 2. 后端基类继承

```
Backend (抽象基类)
    ├── MooncakeBackend
    │   └── Mooncake分布式存储
    │
    ├── MemcacheBackend
    │   └── Memcache内存缓存
    │
    └── YuanrongBackend
        └── Yuanrong远程存储
```

---

## 二、组合关系

### 1. AscendStoreConnector 组合关系

```
AscendStoreConnector
    ├── scheduler: KVPoolScheduler
    │   └── 调度器(检测缓存命中)
    │
    ├── worker: KVPoolWorker
    │   └── 工作器(执行传输)
    │
    └── event_aggregator: EventAggregator
        └── 事件聚合器(管理KV事件)
```

### 2. KVPoolWorker 组合关系

```
KVPoolWorker
    ├── backend: Backend
    │   ├── MooncakeBackend
    │   ├── MemcacheBackend
    │   └── YuanrongBackend
    │
    ├── store_sending_thread: KVCacheStoreSendingThread
    │   └── KV存储发送线程
    │
    ├── store_recving_thread: KVCacheStoreRecvingThread
    │   └── KV存储接收线程
    │
    ├── load_sending_thread: KVCacheLoadSendingThread
    │   └── KV加载发送线程
    │
    └── load_recving_thread: KVCacheLoadRecvingThread
        └── KV加载接收线程
```

### 3. CPUOffloadingConnector 组合关系

```
CPUOffloadingConnector
    ├── connector_scheduler: CPUOffloadingConnectorScheduler
    │   ├── zmq_rpc_client: MetadataServer.ZMQRPCClient
    │   │   └── ZMQ RPC客户端
    │   └── num_gpu_computed_tokens: dict
    │       └── GPU已计算token数
    │
    └── connector_worker: CPUOffloadingConnectorWorker
        ├── zmq_rpc_client: MetadataServer.ZMQRPCClient
        │   └── ZMQ RPC客户端
        ├── gpu_kv_caches: dict
        │   └── GPU KV缓存
        ├── cpu_kv_caches: list
        │   └── CPU KV缓存(共享内存)
        ├── load_stream: torch.npu.Stream
        │   └── 加载流
        ├── save_stream: torch.npu.Stream
        │   └── 保存流
        └── save_thread: threading.Thread
            └── 保存监听线程
```

### 4. MetadataServer 组合关系

```
MetadataServer
    ├── socket: zmq.Socket
    │   └── ZMQ ROUTER socket
    │
    ├── cpu_block_manager: CPUKVCacheManager
    │   ├── block_pool: BlockPool
    │   │   └── 块池管理
    │   ├── single_type_manager: SingleTypeKVCacheManager
    │   │   └── 单类型KV缓存管理器
    │   └── cpu_cache_stats: CPUCacheStats
    │       └── CPU缓存统计
    │
    └── shared_memory: dict
        └── 共享内存字典
```

---

## 三、协作关系

### 1. KV缓存存储流程协作

```
┌─────────────────────────────────────────────────────────────┐
│                     vLLM Scheduler                           │
│  1. request_finished(request)                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              AscendStoreConnector (Scheduler角色)            │
│  2. build_connector_meta(scheduler_output)                  │
│     - 收集请求元数据                                          │
│     - 构建连接器元数据                                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              AscendStoreConnector (Worker角色)               │
│  3. save_kv_layer(layer_name, kv_layer)                     │
│     - 触发保存操作                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   KVPoolWorker                               │
│  4. save_kv_layer(layer_name, kv_layer)                     │
│     - 将KV数据放入保存队列                                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│           KVCacheStoreSendingThread (独立线程)               │
│  5. 从队列获取KV数据                                          │
│  6. 调用 backend.put(keys, addrs, sizes)                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  Backend (Mooncake/Memcache/Yuanrong)        │
│  7. 实际存储KV数据到后端存储系统                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  EventAggregator                             │
│  8. 更新事件状态                                              │
│  9. 通知调度器完成                                            │
└─────────────────────────────────────────────────────────────┘
```

### 2. KV缓存加载流程协作

```
┌─────────────────────────────────────────────────────────────┐
│                     vLLM Scheduler                           │
│  1. 调度新请求                                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              AscendStoreConnector (Scheduler角色)            │
│  2. get_num_new_matched_tokens(request)                     │
│     - 查询KV缓存命中情况                                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   KVPoolScheduler                            │
│  3. lookup_key_client.call("lookup_key", ...)               │
│     - 通过ZMQ RPC查询远程KV缓存                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              AscendStoreConnector (Worker角色)               │
│  4. start_load_kv(forward_context)                          │
│     - 启动KV加载流程                                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   KVPoolWorker                               │
│  5. start_load_kv()                                          │
│     - 启动加载线程                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│           KVCacheLoadSendingThread (独立线程)                │
│  6. 发送加载请求                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│           KVCacheLoadRecvingThread (独立线程)                │
│  7. 调用 backend.get(keys, addrs, sizes)                    │
│  8. 将KV数据加载到GPU内存                                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  Backend (Mooncake/Memcache/Yuanrong)        │
│  9. 从后端存储系统读取KV数据                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              AscendStoreConnector (Worker角色)               │
│  10. wait_for_layer_load(layer_name)                        │
│      - 等待层加载完成                                         │
└─────────────────────────────────────────────────────────────┘
```

### 3. CPU卸载流程协作

```
┌─────────────────────────────────────────────────────────────┐
│                     vLLM Scheduler                           │
│  1. 调度请求                                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│         CPUOffloadingConnector (Scheduler角色)               │
│  2. get_num_new_matched_tokens(request)                     │
│     - 查询CPU缓存命中情况                                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│       CPUOffloadingConnectorScheduler                        │
│  3. zmq_rpc_client.call("get_matched_num_and_touch", ...)   │
│     - 通过ZMQ RPC查询CPU KV缓存                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   MetadataServer (独立进程)                  │
│  4. serve_step() - 处理RPC请求                               │
│  5. cpu_block_manager.get_matched_num_and_touch(request)    │
│     - 查询CPU缓存命中                                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   CPUKVCacheManager                          │
│  6. single_type_manager.find_longest_cache_hit(...)         │
│     - 查找最长缓存命中                                         │
│  7. 返回匹配的token数                                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│         CPUOffloadingConnector (Worker角色)                  │
│  8. start_load_kv()                                          │
│     - 启动从CPU到GPU的加载                                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│       CPUOffloadingConnectorWorker                           │
│  9. load_kv_layer(layer)                                     │
│     - 从CPU共享内存拷贝到GPU内存                               │
│     - 使用独立的NPU流                                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   GPU KV Cache                               │
│  10. KV缓存加载完成                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、数据结构关系

### 1. KV缓存键结构

```
PoolKey (KV池键)
    ├── chunk_hash: str
    │   └── 分块哈希值
    ├── fmt: str
    │   └── 格式标识
    ├── layer_name: str
    │   └── 层名称
    ├── tensor_parallel_rank: int
    │   └── 张量并行秩
    └── pipeline_parallel_rank: int
        └── 流水线并行秩

KeyMetadata (键元数据)
    ├── chunk_hash: str
    ├── fmt: str
    ├── layer_name: str
    ├── tensor_parallel_rank: int
    ├── pipeline_parallel_rank: int
    ├── addrs: list[list[int]]
    │   └── KV数据地址列表
    └── sizes: list[list[int]]
        └── KV数据大小列表
```

### 2. 请求元数据结构

```
ReqMeta (请求元数据 - CPU Offload)
    ├── gpu_block_ids: list[int]
    │   └── GPU块ID列表
    ├── cpu_block_ids: list[int]
    │   └── CPU块ID列表
    ├── num_scheduled_tokens: int
    │   └── 已调度token数
    ├── num_computed_tokens: int
    │   └── 已计算token数
    ├── num_gpu_computed_tokens: int
    │   └── GPU已计算token数
    └── num_cpu_computed_tokens: int
        └── CPU已计算token数

CPUOffloadingConnectorMetadata (连接器元数据)
    ├── requests: dict[str, ReqMeta]
    │   └── 请求ID到元数据的映射
    └── finished_req_ids: set[str]
        └── 已完成请求ID集合
```

### 3. 缓存块结构

```
KVCacheBlock (vLLM)
    ├── block_id: int
    │   └── 块ID
    ├── ref_cnt: int
    │   └── 引用计数
    └── block_hash: BlockHash
        └── 块哈希值

BlockPool (块池)
    ├── free_blocks: list[KVCacheBlock]
    │   └── 空闲块列表
    ├── cached_blocks: dict[BlockHash, KVCacheBlock]
    │   └── 已缓存块字典
    └── block_size: int
        └── 块大小
```

---

## 五、线程模型

### 1. AscendStore 线程模型

```
主线程 (vLLM推理)
    │
    ├── KVCacheStoreSendingThread (守护线程)
    │   ├── 从store_input_queue获取数据
    │   ├── 调用backend.put()存储
    │   └── 更新事件状态
    │
    ├── KVCacheStoreRecvingThread (守护线程)
    │   ├── 从store_output_queue获取结果
    │   └── 更新事件状态
    │
    ├── KVCacheLoadSendingThread (守护线程)
    │   ├── 从load_input_queue获取请求
    │   └── 发送加载请求
    │
    └── KVCacheLoadRecvingThread (守护线程)
        ├── 调用backend.get()加载
        ├── 将数据加载到GPU
        └── 更新事件状态
```

### 2. CPU Offload 线程模型

```
主线程 (vLLM推理)
    │
    ├── MetadataServerProc (独立进程)
    │   ├── MetadataServer
    │   │   ├── serve_step() - 处理RPC请求
    │   │   └── 管理CPU KV缓存
    │   └── CPUKVCacheManager
    │       └── 管理CPU缓存块
    │
    └── save_thread (守护线程)
        ├── 从save_input_queue获取请求
        ├── 使用save_stream (NPU流)
        ├── 从GPU拷贝到CPU共享内存
        └── 将结果放入save_output_queue
```

---

## 六、通信模式

### 1. ZMQ通信模式

```
┌─────────────────────────────────────────────────────────────┐
│              ZMQ DEALER-ROUTER 模式                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  DEALER Socket                ROUTER Socket                 │
│  (Client)                     (Server)                       │
│                                                              │
│  ┌─────────────┐              ┌─────────────┐              │
│  │ Scheduler   │              │ Metadata    │              │
│  │ Worker      │◄────────────►│ Server      │              │
│  └─────────────┘              └─────────────┘              │
│                                                              │
│  特点:                                                       │
│  - 异步通信                                                  │
│  - 支持多客户端                                              │
│  - 自动重连                                                  │
│  - 身份标识                                                  │
└─────────────────────────────────────────────────────────────┘
```

### 2. 共享内存通信模式

```
┌─────────────────────────────────────────────────────────────┐
│              SharedMemory 跨进程通信                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Process 1 (MetadataServer)                                 │
│  ┌─────────────┐                                            │
│  │ 创建        │                                            │
│  │ SharedMemory│                                            │
│  └─────────────┘                                            │
│        ↓                                                     │
│  ┌─────────────┐                                            │
│  │ CPU KV      │                                            │
│  │ Cache       │                                            │
│  └─────────────┘                                            │
│        ↓                                                     │
│  Process 2 (Worker)                                         │
│  ┌─────────────┐                                            │
│  │ 连接        │                                            │
│  │ SharedMemory│                                            │
│  └─────────────┘                                            │
│        ↓                                                     │
│  ┌─────────────┐                                            │
│  │ torch.frombuffer │                                       │
│  │ → Tensor     │                                            │
│  └─────────────┘                                            │
│                                                              │
│  特点:                                                       │
│  - 零拷贝                                                    │
│  - 高性能                                                    │
│  - 跨进程共享                                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 七、配置类关系

```
VllmConfig (vLLM配置)
    ├── cache_config: CacheConfig
    │   ├── block_size: int
    │   ├── cache_dtype: str
    │   └── enable_prefix_caching: bool
    │
    ├── parallel_config: ParallelConfig
    │   ├── world_size: int
    │   ├── tensor_parallel_size: int
    │   └── pipeline_parallel_size: int
    │
    └── kv_transfer_config: KVTransferConfig
        ├── kv_connector: str
        │   ├── "AscendStoreConnector"
        │   ├── "CPUOffloadingConnector"
        │   ├── "LMCacheConnectorV1"
        │   └── "UCMConnectorV1"
        └── extra_config: dict
            ├── mooncake配置
            ├── cpu_offload配置
            └── 其他配置

MooncakeStoreConfig (Mooncake配置)
    ├── metadata_server: str
    ├── global_segment_size: int
    ├── local_buffer_size: int
    ├── protocol: str
    ├── device_name: str
    └── master_server_address: str

YuanrongConfig (Yuanrong配置)
    ├── worker_addr: str
    ├── enable_exclusive_connection: bool
    └── enable_remote_h2d: bool

MLAConfig (MLA配置)
    ├── nope_dim: int
    └── rope_dim: int
```

---

## 总结

本文档详细描述了 `kv_pool` 模块中各个类之间的继承、组合、协作关系，以及数据结构、线程模型和通信模式。通过这些关系图，可以更好地理解整个系统的架构设计和数据流向。
