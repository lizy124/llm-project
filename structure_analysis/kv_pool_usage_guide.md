# KV Pool 使用指南

## 文档说明

本文档记录了KV Pool系统的接入起点和具体使用方式，帮助开发者理解如何在实际项目中使用池化功能。

---

## 核心问题

**用户问题**: 池化系统的接入起点是什么？具体如何使用？

---

## 池化系统概述

池化系统通过**KVTransferConfig配置**作为入口，实现了分布式KV缓存的传输和管理。整个系统采用双角色设计（Scheduler和Worker），支持多种连接器和后端。

---

## 一、配置入口

### 1.1 KVTransferConfig配置类

池化功能通过 "vllm\vllm\config\kv_transfer.py" 中的 `KVTransferConfig` 进行配置：

```python
@config
class KVTransferConfig:
    """KV缓存传输配置"""
    
    kv_connector: str | None = None
    """KV连接器名称，如 'AscendStoreConnector', 'CPUOffloadingConnector' 等"""
    
    kv_role: KVRole | None = None
    """角色类型：'kv_producer'（生产者）、'kv_consumer'（消费者）、'kv_both'（两者兼具）"""
    
    engine_id: str | None = None
    """引擎ID，用于KV传输的唯一标识"""
    
    kv_buffer_device: str = "cuda"
    """KV连接器用于缓冲KV缓存的设备，可选 'cuda', 'cpu', 'xpu'"""
    
    kv_buffer_size: float = 1e9
    """缓冲区大小（字节），推荐值：1e9（约1GB）"""
    
    kv_rank: int | None = None
    """KV缓存传输中的rank，典型值：0表示prefill实例，1表示decode实例"""
    
    kv_parallel_size: int = 1
    """KV缓存传输的并行实例数，P2pNcclConnector应为2"""
    
    kv_ip: str = "127.0.0.1"
    """KV连接器IP地址，用于建立分布式连接"""
    
    kv_port: int = 14579
    """KV连接器端口，用于建立分布式连接"""
    
    kv_connector_extra_config: dict[str, Any] = {}
    """连接器需要的额外配置"""
    
    kv_connector_module_path: str | None = None
    """动态加载KV连接器的Python模块路径（仅V1支持）"""
    
    enable_permute_local_kv: bool = False
    """实验性功能：启用HND到NHD的KV传输"""
    
    kv_load_failure_policy: Literal["recompute", "fail"] = "fail"
    """KV缓存加载失败的处理策略：
    - 'recompute': 重新调度请求以重新计算失败的块
    - 'fail': 立即失败请求并返回错误（默认）
    """
```

### 1.2 角色类型说明

```python
# 生产者：生成KV缓存并传输给其他节点
KVProducer = Literal["kv_producer", "kv_both"]

# 消费者：接收并使用其他节点的KV缓存
KVConsumer = Literal["kv_consumer", "kv_both"]

# 完整角色定义
KVRole = Literal[KVProducer, KVConsumer]
```

### 1.3 便捷属性

```python
@property
def is_kv_transfer_instance(self) -> bool:
    """是否为KV传输实例"""
    return self.kv_connector is not None and self.kv_role in get_args(KVRole)

@property
def is_kv_producer(self) -> bool:
    """是否为KV生产者"""
    return self.kv_connector is not None and self.kv_role in get_args(KVProducer)

@property
def is_kv_consumer(self) -> bool:
    """是否为KV消费者"""
    return self.kv_connector is not None and self.kv_role in get_args(KVConsumer)
```

---

## 二、初始化流程

池化系统的初始化分为**Scheduler端**和**Worker端**两条并行路径。

### 2.1 Scheduler端初始化

#### 步骤1: EngineCore创建Scheduler

**文件**: "vllm\vllm\v1\engine\core.py" (第89-250行)

```python
class EngineCore:
    """vLLM引擎的内部循环"""
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        # 1. 加载插件
        from vllm.plugins import load_general_plugins
        load_general_plugins()
        
        self.vllm_config = vllm_config
        self.log_stats = log_stats
        
        # 2. 设置模型执行器
        self.model_executor = executor_class(vllm_config)
        
        # 3. 初始化KV缓存
        kv_cache_config = self._initialize_kv_caches(vllm_config)
        
        # 4. 创建Scheduler
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        
        # 5. 如果有KVConnector，初始化KV输出聚合器
        if self.scheduler.connector is not None:
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)
        
        # 6. 收集Worker的握手元数据
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # 从所有Worker收集KV连接器传输元数据
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )
            
            if xfer_handshake_metadata:
                # 合并所有Worker的元数据
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)
```

#### 步骤2: Scheduler创建KVConnector

**文件**: "vllm\vllm\v1\core\sched\scheduler.py" (第123-136行)

```python
class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.kv_cache_config = kv_cache_config
        
        # 创建SCHEDULER角色的KVConnector
        # 注意：每个Worker也会有对应的WORKER角色KVConnector
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        
        if self.vllm_config.kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder模型目前不支持KV连接器"
            )
            
            # 通过工厂模式创建连接器
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            
            # 设置KV加载失败策略
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"
```

### 2.2 Worker端初始化

#### 步骤1: GPUWorker初始化KVConnector

**文件**: "vllm\vllm\v1\worker\gpu_worker.py" (第520-550行)

```python
class GPUWorker:
    def initialize(self, ...):
        """初始化Worker"""
        
        # 1. 初始化模型
        self.init_model()
        
        # 2. 初始化KV缓存
        self.initialize_kv_cache(kv_cache_configs)
        
        # 3. 创建WORKER角色的KVConnector
        ensure_kv_transfer_initialized(
            self.vllm_config, kv_cache_config
        )
        
        # 4. 模型预热
        self.warm_up_model()
```

#### 步骤2: 全局KVConnector初始化

**文件**: "vllm\vllm\distributed\kv_transfer\kv_transfer_state.py" (第60-72行)

```python
# 全局KV连接器代理
_KV_CONNECTOR_AGENT: KVConnectorBaseType | None = None


def get_kv_transfer_group() -> KVConnectorBaseType:
    """获取KV传输组"""
    assert _KV_CONNECTOR_AGENT is not None, (
        "分布式KV缓存传输并行组未初始化"
    )
    return _KV_CONNECTOR_AGENT


def has_kv_transfer_group() -> bool:
    """检查KV传输组是否存在"""
    return _KV_CONNECTOR_AGENT is not None


def ensure_kv_transfer_initialized(
    vllm_config: "VllmConfig", 
    kv_cache_config: "KVCacheConfig | None" = None
) -> None:
    """
    初始化KV缓存传输并行组
    """
    global _KV_CONNECTOR_AGENT
    
    # 如果没有配置KV传输，直接返回
    if vllm_config.kv_transfer_config is None:
        return
    
    # 如果是KV传输实例且全局代理未初始化
    if (
        vllm_config.kv_transfer_config.is_kv_transfer_instance
        and _KV_CONNECTOR_AGENT is None
    ):
        # 通过工厂模式创建WORKER角色的连接器
        _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(
            config=vllm_config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=kv_cache_config,
        )


def ensure_kv_transfer_shutdown() -> None:
    """关闭KV传输组"""
    global _KV_CONNECTOR_AGENT
    if _KV_CONNECTOR_AGENT is not None:
        _KV_CONNECTOR_AGENT.shutdown()
        _KV_CONNECTOR_AGENT = None
```

### 2.3 ModelRunner获取连接器

**文件**: "vllm\vllm\v1\worker\gpu\kv_connector.py" (第68-132行)

```python
class ActiveKVConnector(KVConnector):
    """活跃的KV连接器，用于GPUModelRunner"""
    
    def __init__(
        self, 
        vllm_config: VllmConfig, 
        kv_caches_dict: dict[str, torch.Tensor]
    ):
        self.vllm_config = vllm_config
        
        # 获取全局KV连接器
        self.kv_connector = get_kv_transfer_group()
        
        # 向KV连接器注册KV缓存
        # TODO: 支持cross_layers_kv_cache
        # (参见 https://github.com/vllm-project/vllm/pull/27743)
        self.kv_connector.register_kv_caches(kv_caches_dict)
        
        # 设置主机传输缓冲区操作
        self.kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)
        
        self._disabled = False
    
    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        """前向传播前的准备工作"""
        if self._disabled:
            return
        
        # 获取连接器元数据
        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        
        # 绑定元数据
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)
        
        # 处理抢占
        self.kv_connector.handle_preemptions(kv_connector_metadata)
        
        # 启动KV加载
        if is_forward_context_available():
            self.kv_connector.start_load_kv(get_forward_context())
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(get_forward_context())
    
    def post_forward(
        self,
        scheduler_output: "SchedulerOutput",
        wait_for_save: bool = True,
        clear_metadata: bool = True,
    ) -> KVConnectorOutput | None:
        """前向传播后的清理工作"""
        if self._disabled:
            return None
        
        output = KVConnectorOutput()
        
        # 等待保存完成
        if wait_for_save:
            self.kv_connector.wait_for_save()
        
        # 获取完成的请求
        output.finished_sending, output.finished_recving = (
            self.kv_connector.get_finished(scheduler_output.finished_req_ids)
        )
        
        # 获取加载失败的块ID
        output.invalid_block_ids = self.kv_connector.get_block_ids_with_load_errors()
        
        # 获取统计信息
        output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()
        output.kv_cache_events = self.kv_connector.get_kv_connector_kv_cache_events()
        output.kv_connector_worker_meta = (
            self.kv_connector.build_connector_worker_meta()
        )
        
        # 清理元数据
        if clear_metadata:
            self.kv_connector.clear_connector_metadata()
        
        return output
    
    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        """无前向传播时的处理"""
        if self._disabled:
            return EMPTY_MODEL_RUNNER_OUTPUT
        
        self.pre_forward(scheduler_output)
        kv_connector_output = self.post_forward(scheduler_output, wait_for_save=False)
        
        if kv_connector_output is None or kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT
        
        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output
    
    def set_disabled(self, disabled: bool) -> None:
        """设置禁用状态"""
        # 确保禁用时不会调用逐层连接器钩子
        kv_transfer_state._KV_CONNECTOR_AGENT = None if disabled else self.kv_connector
        self._disabled = disabled


# 无操作连接器（当没有KV传输时使用）
NO_OP_KV_CONNECTOR = KVConnector()


def get_kv_connector(
    vllm_config: VllmConfig, 
    kv_caches_dict: dict[str, torch.Tensor]
) -> KVConnector:
    """
    获取KV连接器
    
    Args:
        vllm_config: vLLM配置
        kv_caches_dict: KV缓存字典
        
    Returns:
        KVConnector实例（活跃或无操作）
    """
    if not has_kv_transfer_group():
        # 无操作连接器
        return NO_OP_KV_CONNECTOR
    
    # 返回活跃连接器
    return ActiveKVConnector(vllm_config, kv_caches_dict)
```

---

## 三、具体使用方式

### 3.1 配置示例

#### 示例1: 使用AscendStoreConnector

```python
from vllm.config import KVTransferConfig, EngineArgs
from vllm import LLM

# 配置KVTransferConfig
kv_transfer_config = KVTransferConfig(
    kv_connector="AscendStoreConnector",  # 使用Ascend存储连接器
    kv_role="kv_both",                     # 既生产又消费KV缓存
    kv_buffer_device="cpu",                # 使用CPU作为缓冲设备
    kv_buffer_size=2e9,                    # 2GB缓冲区
    kv_connector_extra_config={
        "use_layerwise": True,             # 启用逐层传输
        "consumer_is_to_put": False,       # consumer不put数据
        "mooncake": {
            "metadata_server": "localhost:12345",
            "protocol": "ascend",
            "device_name": "npu:0"
        }
    }
)

# 创建引擎参数
engine_args = EngineArgs(
    model="your-model-path",
    kv_transfer_config=kv_transfer_config,
    enable_prefix_caching=True,  # 启用前缀缓存
    max_model_len=4096,
    tensor_parallel_size=2,
)

# 创建LLM实例
llm = LLM.from_engine_args(engine_args)
```

#### 示例2: 使用CPUOffloadingConnector

```python
from vllm.config import KVTransferConfig, EngineArgs

# 配置KVTransferConfig
kv_transfer_config = KVTransferConfig(
    kv_connector="SimpleCPUOffloadConnector",  # CPU卸载连接器
    kv_role="kv_both",                          # 既生产又消费
    kv_connector_extra_config={
        "cpu_bytes_to_use": 8 * (1024**3),      # 8GB CPU内存
        "lazy_offload": False,                   # 急切卸载模式
    }
)

# 创建引擎
engine_args = EngineArgs(
    model="your-model-path",
    kv_transfer_config=kv_transfer_config,
    enable_prefix_caching=True,
)

llm = LLM.from_engine_args(engine_args)
```

#### 示例3: 使用P2pNcclConnector（P/D分离）

```python
from vllm.config import KVTransferConfig, EngineArgs

# Prefill实例配置
prefill_config = KVTransferConfig(
    kv_connector="P2pNcclConnector",
    kv_role="kv_producer",  # 生产者
    kv_rank=0,              # rank 0
    kv_parallel_size=2,     # 2个并行实例
    kv_ip="192.168.1.100",
    kv_port=14579,
)

# Decode实例配置
decode_config = KVTransferConfig(
    kv_connector="P2pNcclConnector",
    kv_role="kv_consumer",  # 消费者
    kv_rank=1,              # rank 1
    kv_parallel_size=2,     # 2个并行实例
    kv_ip="192.168.1.100",
    kv_port=14579,
)
```

### 3.2 vllm-ascend的连接器注册

**文件**: "vllm-ascend\vllm_ascend\distributed\kv_transfer\__init__.py"

```python
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory


def register_connector():
    """注册Ascend平台特有的连接器"""
    
    # 覆盖MultiConnector为AscendMultiConnector
    if "MultiConnector" in KVConnectorFactory._registry:
        KVConnectorFactory._registry.pop("MultiConnector")
    
    KVConnectorFactory.register_connector(
        "MultiConnector", 
        "vllm_ascend.distributed.kv_transfer.ascend_multi_connector", 
        "AscendMultiConnector"
    )
    
    # 注册Mooncake连接器
    KVConnectorFactory.register_connector(
        "MooncakeConnectorV1", 
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector", 
        "MooncakeConnector"
    )
    
    # 注册AscendStore连接器
    KVConnectorFactory.register_connector(
        "MooncakeConnectorStoreV1",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
        "AscendStoreConnector",
    )
    
    KVConnectorFactory.register_connector(
        "AscendStoreConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
        "AscendStoreConnector",
    )
    
    # 注册Mooncake逐层连接器
    KVConnectorFactory.register_connector(
        "MooncakeLayerwiseConnector",
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector",
        "MooncakeLayerwiseConnector",
    )
    
    # 注册UCM连接器
    KVConnectorFactory.register_connector(
        "UCMConnector", 
        "vllm_ascend.distributed.kv_transfer.kv_pool.ucm_connector", 
        "UCMConnectorV1"
    )
    
    # 注册LMCache Ascend连接器
    KVConnectorFactory.register_connector(
        "LMCacheAscendConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.lmcache_ascend_connector",
        "LMCacheConnectorV1",
    )
```

### 3.3 运行时调用流程

#### Scheduler端调用流程

```python
# 1. 检查是否有可复用的KV缓存
num_new_matched, has_external = connector.get_num_new_matched_tokens(
    request, 
    num_computed_tokens
)

# 2. 更新请求状态
connector.update_state_after_alloc(
    request, 
    blocks, 
    num_external_tokens
)

# 3. 构建元数据传递给Worker
metadata = connector.build_connector_meta(scheduler_output)

# 4. 请求完成时的处理
should_offload, extra_meta = connector.request_finished(
    request, 
    block_ids
)

# 5. 更新连接器输出
connector.update_connector_output(connector_output)

# 6. 获取KV缓存事件
for event in connector.take_events():
    # 处理事件
    pass
```

#### Worker端调用流程

```python
# 1. 注册KV缓存
connector.register_kv_caches(kv_caches_dict)

# 2. 前向传播前加载KV缓存
def pre_forward(scheduler_output):
    metadata = scheduler_output.kv_connector_metadata
    connector.bind_connector_metadata(metadata)
    connector.handle_preemptions(metadata)
    connector.start_load_kv(forward_context)

# 3. 前向传播中逐层保存KV缓存（如果启用layerwise）
def save_kv_layer(layer_name, kv_layer, attn_metadata):
    connector.save_kv_layer(layer_name, kv_layer, attn_metadata)

# 4. 前向传播后等待保存完成
def post_forward(scheduler_output):
    connector.wait_for_save()
    finished_sending, finished_recving = connector.get_finished(
        scheduler_output.finished_req_ids
    )
    stats = connector.get_kv_connector_stats()
    events = connector.get_kv_connector_kv_cache_events()
    return KVConnectorOutput(
        finished_sending=finished_sending,
        finished_recving=finished_recving,
        kv_connector_stats=stats,
        kv_cache_events=events,
    )
```

---

## 四、关键设计点

### 4.1 双角色设计

池化系统采用双角色设计，分离调度和执行：

| 角色 | 职责 | 创建位置 | 主要方法 |
|------|------|---------|---------|
| **SCHEDULER** | 调度、元数据管理、匹配逻辑 | EngineCore → Scheduler | `get_num_new_matched_tokens()`<br>`update_state_after_alloc()`<br>`build_connector_meta()` |
| **WORKER** | 实际的KV缓存传输操作 | GPUWorker → ensure_kv_transfer_initialized() | `register_kv_caches()`<br>`start_load_kv()`<br>`save_kv_layer()` |

### 4.2 插件化架构

通过工厂模式实现插件化：

```python
# 注册新连接器
KVConnectorFactory.register_connector(
    "MyCustomConnector",
    "my_module.my_connector",
    "MyConnector"
)

# 创建连接器
connector = KVConnectorFactory.create_connector(
    config=vllm_config,
    role=KVConnectorRole.SCHEDULER,
    kv_cache_config=kv_cache_config,
)
```

### 4.3 异步传输机制

支持异步传输以避免阻塞推理流程：

```python
# pre_forward阶段：启动异步加载
connector.start_load_kv(forward_context)

# 执行模型前向传播
output = model(input_ids)

# post_forward阶段：等待保存完成
connector.wait_for_save()
```

### 4.4 逐层传输优化

对于支持逐层传输的连接器，可以在每层完成后立即传输：

```python
for layer_name, layer in model.layers:
    # 执行层计算
    hidden_states = layer(hidden_states)
    
    # 立即保存该层的KV缓存
    if kv_connector is not None:
        kv_connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
        kv_connector.wait_for_layer_load(layer_name)
```

---

## 五、完整示例

### 5.1 分布式推理场景（P/D分离）

```python
# ===== Prefill实例 =====
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig, EngineArgs

# 配置Prefill实例
prefill_config = KVTransferConfig(
    kv_connector="P2pNcclConnector",
    kv_role="kv_producer",
    kv_rank=0,
    kv_parallel_size=2,
    kv_ip="192.168.1.100",
    kv_port=14579,
)

prefill_args = EngineArgs(
    model="your-model",
    kv_transfer_config=prefill_config,
    enable_prefix_caching=True,
)

prefill_llm = LLM.from_engine_args(prefill_args)

# 执行prefill
prompts = ["Hello, world!"] * 10
outputs = prefill_llm.generate(prompts, SamplingParams(max_tokens=1))


# ===== Decode实例 =====
# 配置Decode实例
decode_config = KVTransferConfig(
    kv_connector="P2pNcclConnector",
    kv_role="kv_consumer",
    kv_rank=1,
    kv_parallel_size=2,
    kv_ip="192.168.1.100",
    kv_port=14579,
)

decode_args = EngineArgs(
    model="your-model",
    kv_transfer_config=decode_config,
    enable_prefix_caching=True,
)

decode_llm = LLM.from_engine_args(decode_args)

# 执行decode（会自动接收KV缓存）
outputs = decode_llm.generate(prompts, SamplingParams(max_tokens=100))
```

### 5.2 CPU卸载场景

```python
from vllm import LLM
from vllm.config import KVTransferConfig, EngineArgs

# 配置CPU卸载
kv_transfer_config = KVTransferConfig(
    kv_connector="SimpleCPUOffloadConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "cpu_bytes_to_use": 8 * (1024**3),  # 8GB
        "lazy_offload": False,
    }
)

engine_args = EngineArgs(
    model="your-model",
    kv_transfer_config=kv_transfer_config,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.9,
)

llm = LLM.from_engine_args(engine_args)

# 正常使用，KV缓存会自动在GPU和CPU之间交换
outputs = llm.generate(prompts)
```

### 5.3 Ascend Store场景

```python
from vllm import LLM
from vllm.config import KVTransferConfig, EngineArgs

# 配置Ascend Store
kv_transfer_config = KVTransferConfig(
    kv_connector="AscendStoreConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "use_layerwise": True,
        "consumer_is_to_put": False,
        "mooncake": {
            "metadata_server": "localhost:12345",
            "protocol": "ascend",
            "device_name": "npu:0"
        }
    }
)

engine_args = EngineArgs(
    model="your-model",
    kv_transfer_config=kv_transfer_config,
    enable_prefix_caching=True,
    tensor_parallel_size=2,
)

llm = LLM.from_engine_args(engine_args)

# 正常使用，KV缓存会自动存储到分布式存储系统
outputs = llm.generate(prompts)
```

---

## 六、常见问题

### Q1: 如何选择连接器？

| 连接器 | 适用场景 | 优势 | 劣势 |
|--------|---------|------|------|
| **AscendStoreConnector** | 分布式推理，跨节点共享KV缓存 | 支持多种后端，高性能RDMA传输 | 需要配置存储后端 |
| **SimpleCPUOffloadConnector** | 单节点显存不足 | 简单易用，无需额外依赖 | CPU-GPU传输开销 |
| **P2pNcclConnector** | P/D分离场景 | 低延迟，高性能 | 仅支持1P1D拓扑 |
| **LMCacheConnectorV1** | 已有LMCache基础设施 | 兼容现有系统 | 依赖LMCache |
| **UCMConnectorV1** | 特定UCM场景 | 定制化优化 | 适用范围有限 |

### Q2: 如何选择后端？

| 后端 | 适用场景 | 性能 | 持久化 |
|------|---------|------|--------|
| **Mooncake** | RDMA环境，高性能要求 | ⭐⭐⭐⭐⭐ | ❌ |
| **Memcache** | 内存缓存，快速访问 | ⭐⭐⭐⭐ | ❌ |
| **Yuanrong** | 远程存储，持久化需求 | ⭐⭐⭐ | ✅ |

### Q3: 如何监控性能？

```python
# 1. 查看缓存命中率
stats = connector.get_kv_connector_stats()
print(f"Cache hit rate: {stats.cache_hit_rate}")

# 2. 监控传输延迟
print(f"Load latency: {stats.load_latency_ms}ms")
print(f"Save latency: {stats.save_latency_ms}ms")

# 3. 统计GPU/CPU利用率
import psutil
import GPUtil
print(f"GPU utilization: {GPUtil.getGPUs()[0].load * 100}%")
print(f"CPU utilization: {psutil.cpu_percent()}%")

# 4. 分析事件等待时间
for event in connector.take_events():
    print(f"Event: {event.type}, duration: {event.duration_ms}ms")
```

### Q4: 如何处理加载失败？

```python
# 配置失败策略
kv_transfer_config = KVTransferConfig(
    kv_connector="AscendStoreConnector",
    kv_role="kv_consumer",
    kv_load_failure_policy="recompute",  # 或 "fail"
)

# 如果设置为 "recompute"，失败的块会被重新计算
# 如果设置为 "fail"，请求会立即失败
```

---

## 七、性能优化建议

### 7.1 启用前缀缓存

```python
engine_args = EngineArgs(
    model="your-model",
    enable_prefix_caching=True,  # 启用前缀缓存
    kv_transfer_config=kv_transfer_config,
)
```

### 7.2 选择合适的缓冲区大小

```python
# 根据模型大小和并发量调整
# 推荐：模型参数量 × 2
kv_transfer_config = KVTransferConfig(
    kv_buffer_size=2e9,  # 2GB
)
```

### 7.3 启用异步传输

```python
# 使用逐层传输避免阻塞
kv_transfer_config = KVTransferConfig(
    kv_connector="AscendStoreConnector",
    kv_connector_extra_config={
        "use_layerwise": True,  # 启用逐层传输
    }
)
```

### 7.4 优化线程数

```python
# 根据硬件配置调整传输线程数
kv_transfer_config = KVTransferConfig(
    kv_connector_extra_config={
        "num_transfer_threads": 8,  # 传输线程数
    }
)
```

---

## 八、总结

### 8.1 核心要点

1. **配置入口**: 通过 `KVTransferConfig` 配置连接器、角色、缓冲区等参数
2. **双角色设计**: Scheduler负责调度，Worker负责执行
3. **插件化架构**: 通过工厂模式注册和创建连接器
4. **异步传输**: 支持逐层传输优化性能
5. **多后端支持**: 支持Mooncake、Memcache、Yuanrong等多种后端

### 8.2 使用流程

```
1. 配置 KVTransferConfig
   ↓
2. 创建 EngineArgs
   ↓
3. 初始化 Engine
   ├─ Scheduler创建SCHEDULER角色连接器
   └─ Worker创建WORKER角色连接器
   ↓
4. 运行时调用
   ├─ Scheduler: 匹配、调度、元数据管理
   └─ Worker: 加载、保存、传输KV缓存
```

### 8.3 最佳实践

- ✅ 启用前缀缓存以提高命中率
- ✅ 根据场景选择合适的连接器和后端
- ✅ 调整缓冲区大小以平衡性能和内存
- ✅ 启用异步传输以避免阻塞
- ✅ 监控性能指标以优化配置

---

## 参考资料

- "analysis\llm-project\structure_analysis\kv_pool_quick_overview.md"
- "analysis\llm-project\structure_analysis\kv_pool_analysis.md"
- "analysis\llm-project\structure_analysis\kv_pool_api_reference.md"
- "analysis\llm-project\structure_analysis\kv_pool_class_relationships.md"

---

**文档版本**: 1.0  
**最后更新**: 2024年  
**作者**: vLLM-Ascend分析团队
