# KV Pool API 参考

## 一、连接器 API

### 1. AscendStoreConnector

**位置**: `ascend_store/ascend_store_connector.py`

**继承**: `KVConnectorBase_V1`

#### 主要方法

##### Scheduler 角色方法

```python
def get_num_new_matched_tokens(
    self, 
    request: Request
) -> Optional[int]
```
**功能**: 获取请求的新匹配token数（KV缓存命中）
**参数**: 
- `request`: vLLM请求对象
**返回**: 匹配的token数，如果没有匹配则返回None

```python
def build_connector_meta(
    self, 
    scheduler_output: SchedulerOutput
) -> AscendConnectorMetadata
```
**功能**: 构建连接器元数据
**参数**:
- `scheduler_output`: 调度器输出
**返回**: AscendConnectorMetadata对象

```python
def update_state(
    self, 
    scheduler_output: SchedulerOutput
) -> None
```
**功能**: 更新调度器状态
**参数**:
- `scheduler_output`: 调度器输出

##### Worker 角色方法

```python
def save_kv_layer(
    self, 
    layer_name: str, 
    kv_layer: torch.Tensor
) -> None
```
**功能**: 保存KV缓存层到后端存储
**参数**:
- `layer_name`: 层名称
- `kv_layer`: KV缓存张量

```python
def start_load_kv(
    self, 
    forward_context: ForwardContext
) -> None
```
**功能**: 启动KV缓存加载流程
**参数**:
- `forward_context`: 前向上下文

```python
def wait_for_layer_load(
    self, 
    layer_name: str
) -> None
```
**功能**: 等待指定层的KV缓存加载完成
**参数**:
- `layer_name`: 层名称

---

### 2. CPUOffloadingConnector

**位置**: `cpu_offload/cpu_offload_connector.py`

**继承**: `KVConnectorBase_V1`

#### 主要方法

##### Scheduler 角色方法

```python
def get_num_new_matched_tokens(
    self, 
    request: Request
) -> Optional[int]
```
**功能**: 获取CPU缓存中匹配的token数
**参数**:
- `request`: vLLM请求对象
**返回**: 匹配的token数

```python
def build_connector_meta(
    self, 
    scheduler_output: SchedulerOutput
) -> CPUOffloadingConnectorMetadata
```
**功能**: 构建CPU卸载连接器元数据
**参数**:
- `scheduler_output`: 调度器输出
**返回**: CPUOffloadingConnectorMetadata对象

##### Worker 角色方法

```python
def start_load_kv(
    self, 
    forward_context: ForwardContext
) -> None
```
**功能**: 启动从CPU到GPU的KV缓存加载
**参数**:
- `forward_context`: 前向上下文

```python
def save_kv_layer(
    self, 
    layer_name: str, 
    kv_layer: torch.Tensor
) -> None
```
**功能**: 保存KV缓存层到CPU内存
**参数**:
- `layer_name`: 层名称
- `kv_layer`: KV缓存张量

```python
def wait_for_layer_load(
    self, 
    layer_name: str
) -> None
```
**功能**: 等待层加载完成
**参数**:
- `layer_name`: 层名称

---

### 3. LMCacheConnectorV1

**位置**: `lmcache_ascend_connector.py`

**继承**: `KVConnectorBase_V1`

#### 主要方法

```python
def get_num_new_matched_tokens(
    self, 
    request: Request
) -> Optional[int]
```
**功能**: 获取LMCache中匹配的token数
**参数**:
- `request`: vLLM请求对象
**返回**: 匹配的token数

```python
def save_kv_layer(
    self, 
    layer_name: str, 
    kv_layer: torch.Tensor
) -> None
```
**功能**: 保存KV缓存层到LMCache
**参数**:
- `layer_name`: 层名称
- `kv_layer`: KV缓存张量

---

### 4. UCMConnectorV1

**位置**: `ucm_connector.py`

**继承**: `KVConnectorBase_V1`

#### 主要方法

```python
def get_num_new_matched_tokens(
    self, 
    request: Request
) -> Optional[int]
```
**功能**: 获取UCM中匹配的token数
**参数**:
- `request`: vLLM请求对象
**返回**: 匹配的token数

```python
def save_kv_layer(
    self, 
    layer_name: str, 
    kv_layer: torch.Tensor
) -> None
```
**功能**: 保存KV缓存层到UCM
**参数**:
- `layer_name`: 层名称
- `kv_layer`: KV缓存张量

```python
def start_load_kv(
    self, 
    forward_context: ForwardContext
) -> None
```
**功能**: 启动KV缓存加载
**参数**:
- `forward_context`: 前向上下文

```python
def wait_for_layer_load(
    self, 
    layer_name: str
) -> None
```
**功能**: 等待层加载完成
**参数**:
- `layer_name`: 层名称

---

## 二、后端 API

### 1. Backend (抽象基类)

**位置**: `ascend_store/backend/backend.py`

#### 抽象方法

```python
def __init__(
    self, 
    vllm_config: VllmConfig, 
    **kwargs
) -> None
```
**功能**: 初始化后端
**参数**:
- `vllm_config`: vLLM配置对象
- `**kwargs`: 额外参数

```python
def setup_device(
    self, 
    device: torch.device
) -> None
```
**功能**: 设置设备
**参数**:
- `device`: PyTorch设备对象

```python
def register_buffer(
    self, 
    buffer: torch.Tensor
) -> None
```
**功能**: 注册缓冲区
**参数**:
- `buffer`: 要注册的张量缓冲区

```python
def put(
    self, 
    keys: list[str], 
    addrs: list[list[int]], 
    sizes: list[list[int]]
) -> None
```
**功能**: 存储KV数据
**参数**:
- `keys`: 键列表
- `addrs`: 地址列表
- `sizes`: 大小列表

```python
def get(
    self, 
    keys: list[str], 
    addrs: list[list[int]], 
    sizes: list[list[int]]
) -> None
```
**功能**: 获取KV数据
**参数**:
- `keys`: 键列表
- `addrs`: 地址列表
- `sizes`: 大小列表

---

### 2. MooncakeBackend

**位置**: `ascend_store/backend/mooncake_backend.py`

**继承**: `Backend`

#### 主要方法

```python
def __init__(
    self, 
    vllm_config: VllmConfig, 
    **kwargs
) -> None
```
**功能**: 初始化Mooncake后端
**参数**:
- `vllm_config`: vLLM配置对象
- `**kwargs`: 额外参数（包含mooncake配置）

```python
def setup_device(
    self, 
    device: torch.device
) -> None
```
**功能**: 设置NPU设备并初始化传输对象
**参数**:
- `device`: NPU设备对象

```python
def register_buffer(
    self, 
    buffer: torch.Tensor
) -> None
```
**功能**: 向Mooncake注册GPU缓冲区
**参数**:
- `buffer`: GPU缓冲区张量

```python
def put(
    self, 
    keys: list[str], 
    addrs: list[list[int]], 
    sizes: list[list[int]]
) -> None
```
**功能**: 使用Mooncake存储KV数据
**参数**:
- `keys`: 键列表
- `addrs`: 地址列表
- `sizes`: 大小列表

```python
def get(
    self, 
    keys: list[str], 
    addrs: list[list[int]], 
    sizes: list[list[int]]
) -> None
```
**功能**: 使用Mooncake获取KV数据
**参数**:
- `keys`: 键列表
- `addrs`: 地址列表
- `sizes`: 大小列表

---

### 3. MemcacheBackend

**位置**: `ascend_store/backend/memcache_backend.py`

**继承**: `Backend`

#### 主要方法

```python
def __init__(
    self, 
    vllm_config: VllmConfig, 
    **kwargs
) -> None
```
**功能**: 初始化Memcache后端
**参数**:
- `vllm_config`: vLLM配置对象
- `**kwargs`: 额外参数

```python
def setup_device(
    self, 
    device: torch.device
) -> None
```
**功能**: 设置设备（支持A2设备）
**参数**:
- `device`: 设备对象

```python
def register_buffer(
    self, 
    buffer: torch.Tensor
) -> None
```
**功能**: 注册缓冲区到Memcache
**参数**:
- `buffer`: 缓冲区张量

```python
def put(
    self, 
    keys: list[str], 
    addrs: list[list[int]], 
    sizes: list[list[int]]
) -> None
```
**功能**: 使用Memcache存储KV数据
**参数**:
- `keys`: 键列表
- `addrs`: 地址列表
- `sizes`: 大小列表

```python
def get(
    self, 
    keys: list[str], 
    addrs: list[list[int]], 
    sizes: list[list[int]]
) -> None
```
**功能**: 使用Memcache获取KV数据
**参数**:
- `keys`: 键列表
- `addrs`: 地址列表
- `sizes`: 大小列表

---

### 4. YuanrongBackend

**位置**: `ascend_store/backend/yuanrong_backend.py`

**继承**: `Backend`

#### 主要方法

```python
def __init__(
    self, 
    vllm_config: VllmConfig, 
    **kwargs
) -> None
```
**功能**: 初始化Yuanrong后端
**参数**:
- `vllm_config`: vLLM配置对象
- `**kwargs`: 额外参数（包含yuanrong配置）

```python
def setup_device(
    self, 
    device: torch.device
) -> None
```
**功能**: 设置设备
**参数**:
- `device`: 设备对象

```python
def register_buffer(
    self, 
    buffer: torch.Tensor
) -> None
```
**功能**: 注册缓冲区
**参数**:
- `buffer`: 缓冲区张量

```python
def put(
    self, 
    keys: list[str], 
    addrs: list[list[int]], 
    sizes: list[list[int]]
) -> None
```
**功能**: 使用Yuanrong存储KV数据
**参数**:
- `keys`: 键列表
- `addrs`: 地址列表
- `sizes`: 大小列表

```python
def get(
    self, 
    keys: list[str], 
    addrs: list[list[int]], 
    sizes: list[list[int]]
) -> None
```
**功能**: 使用Yuanrong获取KV数据
**参数**:
- `keys`: 键列表
- `addrs`: 地址列表
- `sizes`: 大小列表

---

## 三、调度器 API

### 1. KVPoolScheduler

**位置**: `ascend_store/pool_scheduler.py`

#### 主要方法

```python
def __init__(
    self, 
    vllm_config: VllmConfig, 
    is_driver_worker: bool
) -> None
```
**功能**: 初始化调度器
**参数**:
- `vllm_config`: vLLM配置对象
- `is_driver_worker`: 是否为驱动worker

```python
def update_state(
    self, 
    scheduler_output: SchedulerOutput
) -> None
```
**功能**: 更新调度器状态
**参数**:
- `scheduler_output`: 调度器输出

```python
def get_num_new_matched_tokens(
    self, 
    request: Request
) -> Optional[int]
```
**功能**: 获取新匹配的token数
**参数**:
- `request`: 请求对象
**返回**: 匹配的token数

```python
def build_connector_meta(
    self, 
    scheduler_output: SchedulerOutput
) -> AscendConnectorMetadata
```
**功能**: 构建连接器元数据
**参数**:
- `scheduler_output`: 调度器输出
**返回**: AscendConnectorMetadata对象

```python
def request_finished(
    self, 
    request_id: str
) -> None
```
**功能**: 处理请求完成事件
**参数**:
- `request_id`: 请求ID

---

### 2. CPUOffloadingConnectorScheduler

**位置**: `cpu_offload/cpu_offload_connector.py`

#### 主要方法

```python
def __init__(
    self, 
    vllm_config: VllmConfig, 
    is_driver_worker: bool
) -> None
```
**功能**: 初始化CPU卸载调度器
**参数**:
- `vllm_config`: vLLM配置对象
- `is_driver_worker`: 是否为驱动worker

```python
def get_num_new_matched_tokens(
    self, 
    request: Request
) -> Optional[int]
```
**功能**: 获取CPU缓存中匹配的token数
**参数**:
- `request`: 请求对象
**返回**: 匹配的token数

```python
def build_connector_meta(
    self, 
    scheduler_output: SchedulerOutput
) -> CPUOffloadingConnectorMetadata
```
**功能**: 构建CPU卸载连接器元数据
**参数**:
- `scheduler_output`: 调度器输出
**返回**: CPUOffloadingConnectorMetadata对象

---

## 四、工作器 API

### 1. KVPoolWorker

**位置**: `ascend_store/pool_worker.py`

#### 主要方法

```python
def __init__(
    self, 
    vllm_config: VllmConfig, 
    is_driver_worker: bool
) -> None
```
**功能**: 初始化工作器
**参数**:
- `vllm_config`: vLLM配置对象
- `is_driver_worker`: 是否为驱动worker

```python
def register_backend(
    self, 
    backend: Backend
) -> None
```
**功能**: 注册后端
**参数**:
- `backend`: 后端对象

```python
def save_kv_layer(
    self, 
    layer_name: str, 
    kv_layer: torch.Tensor
) -> None
```
**功能**: 保存KV缓存层
**参数**:
- `layer_name`: 层名称
- `kv_layer`: KV缓存张量

```python
def start_load_kv(
    self, 
    forward_context: ForwardContext
) -> None
```
**功能**: 启动KV缓存加载
**参数**:
- `forward_context`: 前向上下文

```python
def wait_for_layer_load(
    self, 
    layer_name: str
) -> None
```
**功能**: 等待层加载完成
**参数**:
- `layer_name`: 层名称

```python
def close(self) -> None
```
**功能**: 关闭工作器并清理资源

---

### 2. CPUOffloadingConnectorWorker

**位置**: `cpu_offload/cpu_offload_connector.py`

#### 主要方法

```python
def __init__(
    self, 
    vllm_config: VllmConfig, 
    is_driver_worker: bool
) -> None
```
**功能**: 初始化CPU卸载工作器
**参数**:
- `vllm_config`: vLLM配置对象
- `is_driver_worker`: 是否为驱动worker

```python
def save_kv_layer(
    self, 
    layer_name: str, 
    kv_layer: torch.Tensor
) -> None
```
**功能**: 保存KV缓存层到CPU内存
**参数**:
- `layer_name`: 层名称
- `kv_layer`: KV缓存张量

```python
def start_load_kv(
    self, 
    forward_context: ForwardContext
) -> None
```
**功能**: 启动从CPU到GPU的KV缓存加载
**参数**:
- `forward_context`: 前向上下文

```python
def wait_for_layer_load(
    self, 
    layer_name: str
) -> None
```
**功能**: 等待层加载完成
**参数**:
- `layer_name`: 层名称

```python
def close(self) -> None
```
**功能**: 关闭工作器并清理资源

---

## 五、元数据管理 API

### 1. MetadataServer

**位置**: `cpu_offload/metadata.py`

#### 主要方法

```python
def __init__(
    self, 
    vllm_config: VllmConfig
) -> None
```
**功能**: 初始化元数据服务器
**参数**:
- `vllm_config`: vLLM配置对象

```python
def create_cpu_kv_cache(
    self, 
    kv_cache_config: KVCacheConfig
) -> None
```
**功能**: 创建CPU KV缓存
**参数**:
- `kv_cache_config`: KV缓存配置

```python
def serve_step(self) -> None
```
**功能**: 执行一步服务循环，处理RPC请求

```python
def close(self) -> None
```
**功能**: 关闭服务器并清理资源

---

### 2. CPUKVCacheManager

**位置**: `cpu_offload/cpu_kv_cache_manager.py`

#### 主要方法

```python
def __init__(
    self, 
    block_size: int, 
    num_blocks: int, 
    kv_cache_shape: tuple, 
    dtype: torch.dtype
) -> None
```
**功能**: 初始化CPU KV缓存管理器
**参数**:
- `block_size`: 块大小
- `num_blocks`: 块数量
- `kv_cache_shape`: KV缓存形状
- `dtype`: 数据类型

```python
def get_matched_num_and_touch(
    self, 
    request: Request
) -> tuple[int, list[int]]
```
**功能**: 获取匹配的token数并更新访问时间
**参数**:
- `request`: 请求对象
**返回**: (匹配token数, CPU块ID列表)

```python
def allocate_cpu_blocks(
    self, 
    request: Request
) -> list[int]
```
**功能**: 为请求分配CPU块
**参数**:
- `request`: 请求对象
**返回**: CPU块ID列表

```python
def free_cpu_blocks(
    self, 
    block_ids: list[int]
) -> None
```
**功能**: 释放CPU块
**参数**:
- `block_ids`: 要释放的块ID列表

```python
def get_stats(self) -> CPUCacheStats
```
**功能**: 获取缓存统计信息
**返回**: CPUCacheStats对象

---

## 六、数据类 API

### 1. PoolKey

**位置**: `ascend_store/config_data.py`

```python
@dataclass
class PoolKey:
    chunk_hash: str
    fmt: str
    layer_name: str
    tensor_parallel_rank: int
    pipeline_parallel_rank: int
```

**功能**: KV池键，用于唯一标识KV缓存

---

### 2. KeyMetadata

**位置**: `ascend_store/config_data.py`

```python
@dataclass
class KeyMetadata:
    chunk_hash: str
    fmt: str
    layer_name: str
    tensor_parallel_rank: int
    pipeline_parallel_rank: int
    addrs: list[list[int]]
    sizes: list[list[int]]
```

**功能**: 键元数据，包含KV缓存的地址和大小信息

---

### 3. ReqMeta

**位置**: `cpu_offload/cpu_offload_connector.py`

```python
@dataclass
class ReqMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]
    num_scheduled_tokens: int
    num_computed_tokens: int
    num_gpu_computed_tokens: int
    num_cpu_computed_tokens: int
```

**功能**: 请求元数据，用于CPU卸载

---

### 4. CPUOffloadingConnectorMetadata

**位置**: `cpu_offload/cpu_offload_connector.py`

```python
@dataclass
class CPUOffloadingConnectorMetadata:
    requests: dict[str, ReqMeta]
    finished_req_ids: set[str]
```

**功能**: CPU卸载连接器元数据

---

## 七、配置类 API

### 1. MooncakeStoreConfig

**位置**: `ascend_store/backend/mooncake_backend.py`

```python
@dataclass
class MooncakeStoreConfig:
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
```

**功能**: Mooncake存储配置

---

### 2. YuanrongConfig

**位置**: `ascend_store/backend/yuanrong_backend.py`

```python
@dataclass
class YuanrongConfig:
    worker_addr: str
    enable_exclusive_connection: bool
    enable_remote_h2d: bool
```

**功能**: Yuanrong后端配置

---

### 3. MLAConfig

**位置**: `cpu_offload/metadata.py`

```python
@dataclass
class MLAConfig:
    nope_dim: int
    rope_dim: int
```

**功能**: MLA（Multi-Head Latent Attention）配置

---

## 八、事件管理 API

### 1. AscendStoreKVEvents

**位置**: `ascend_store/ascend_store_connector.py`

#### 主要方法

```python
def __init__(self) -> None
```
**功能**: 初始化事件管理器

```python
def create_event(
    self, 
    event_id: str
) -> None
```
**功能**: 创建事件
**参数**:
- `event_id`: 事件ID

```python
def set_event(
    self, 
    event_id: str
) -> None
```
**功能**: 设置事件状态为完成
**参数**:
- `event_id`: 事件ID

```python
def wait_event(
    self, 
    event_id: str
) -> None
```
**功能**: 等待事件完成
**参数**:
- `event_id`: 事件ID

---

## 九、线程类 API

### 1. KVCacheStoreSendingThread

**位置**: `ascend_store/kv_transfer.py`

#### 主要方法

```python
def __init__(
    self, 
    backend: Backend, 
    input_queue: Queue, 
    output_queue: Queue, 
    event_aggregator: EventAggregator
) -> None
```
**功能**: 初始化存储发送线程
**参数**:
- `backend`: 后端对象
- `input_queue`: 输入队列
- `output_queue`: 输出队列
- `event_aggregator`: 事件聚合器

```python
def run(self) -> None
```
**功能**: 线程主循环，从队列获取数据并调用后端存储

---

### 2. KVCacheLoadRecvingThread

**位置**: `ascend_store/kv_transfer.py`

#### 主要方法

```python
def __init__(
    self, 
    backend: Backend, 
    input_queue: Queue, 
    output_queue: Queue, 
    event_aggregator: EventAggregator
) -> None
```
**功能**: 初始化加载接收线程
**参数**:
- `backend`: 后端对象
- `input_queue`: 输入队列
- `output_queue`: 输出队列
- `event_aggregator`: 事件聚合器

```python
def run(self) -> None
```
**功能**: 线程主循环，调用后端加载数据到GPU

---

## 十、使用示例

### 1. 使用AscendStoreConnector

```python
from vllm.config import VllmConfig
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import AscendStoreConnector

# 创建配置
vllm_config = VllmConfig(...)
vllm_config.kv_transfer_config.kv_connector = "AscendStoreConnector"
vllm_config.kv_transfer_config.extra_config = {
    "mooncake": {
        "metadata_server": "localhost:12345",
        "protocol": "ascend",
        "device_name": "npu:0"
    }
}

# 创建连接器
connector = AscendStoreConnector(vllm_config, is_driver_worker=True)

# Scheduler角色：查询缓存命中
matched_tokens = connector.get_num_new_matched_tokens(request)

# Worker角色：保存KV缓存
connector.save_kv_layer("layer.0", kv_tensor)

# Worker角色：加载KV缓存
connector.start_load_kv(forward_context)
connector.wait_for_layer_load("layer.0")

# 清理
connector.close()
```

### 2. 使用CPUOffloadingConnector

```python
from vllm.config import VllmConfig
from vllm_ascend.distributed.kv_transfer.kv_pool.cpu_offload import CPUOffloadingConnector

# 创建配置
vllm_config = VllmConfig(...)
vllm_config.kv_transfer_config.kv_connector = "CPUOffloadingConnector"
vllm_config.kv_transfer_config.extra_config = {
    "cpu_offload": {
        "buffer_size": 1024 * 1024 * 1024  # 1GB
    }
}

# 创建连接器
connector = CPUOffloadingConnector(vllm_config, is_driver_worker=True)

# Scheduler角色：查询CPU缓存命中
matched_tokens = connector.get_num_new_matched_tokens(request)

# Worker角色：保存到CPU
connector.save_kv_layer("layer.0", kv_tensor)

# Worker角色：从CPU加载
connector.start_load_kv(forward_context)
connector.wait_for_layer_load("layer.0")

# 清理
connector.close()
```

---

## 总结

本文档提供了 `kv_pool` 模块中所有主要类和方法的API参考，包括：
- 连接器API（AscendStore、CPUOffload、LMCache、UCM）
- 后端API（Mooncake、Memcache、Yuanrong）
- 调度器和工作器API
- 元数据管理API
- 数据类和配置类API
- 事件管理和线程类API
- 使用示例

通过这些API，开发者可以灵活地使用和扩展KV缓存传输功能。
