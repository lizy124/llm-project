# PD 分离与池化：概念与代码层面的区分

> **参考文档：** 官方设计文档 [KV_Cache_Pool_Guide.md](../docs/source/developer_guide/Design_Documents/KV_Cache_Pool_Guide.md) / [disaggregated_prefill.md](../docs/source/developer_guide/Design_Documents/disaggregated_prefill.md)

## 零、先搞清楚：Mooncake 是什么

Mooncake 是底层基础设施，PD 分离和池化都用它，但用的是不同能力：

| | PD 分离用的 Mooncake 能力 | 池化用的 Mooncake 能力 |
|---|---|---|
| **组件** | `TransferEngine`（P2P 传输引擎） | Mooncake Store（KV 存储 + master 服务） |
| **用途** | P2P 直传，不落地 | 持久化存储，按 key 查找加载 |
| **数据生命周期** | 一次性传输，用完即丢 | 持久化，跨请求复用 |

**两者可以共存：** 通过 `MultiConnector` 同时启用池化 + PD 分离——池化负责 prefix cache 复用，PD 分离负责 P2P 传输。

---

## 一、目录层面的证据

`vllm_ascend/distributed/kv_transfer/` 下有两个平级子目录：

```
kv_transfer/
├── kv_p2p/          ← PD 分离（prefill→decode 直传）
│   ├── mooncake_connector.py              (MooncakeConnectorV1)
│   ├── mooncake_layerwise_connector.py    (MooncakeLayerwiseConnector)
│   └── mooncake_hybrid_connector.py       (MooncakeHybridConnector)
└── kv_pool/         ← 池化（存储→查找→复用）
    ├── ascend_store/                      (AscendStoreConnector)
    ├── simple_cpu_offload/                (SimpleCPUOffloadConnector)
    ├── recompute_cpu_offload/             (RecomputeCPUOffloadConnector)
    ├── lmcache_ascend_connector.py        (LMCacheAscendConnector)
    └── ucm_connector.py                   (UCMConnector)
```

两者是 `kv_transfer` 下的两个独立子模块，职责不同，没有父子或从属关系。

---

## 二、PD 分离（kv_p2p）：prefill → decode 直传

### 2.1 核心机制

PD 分离的本质是：**prefill 节点算完 KV cache 后，直接 P2P 传给 decode 节点**。数据流是单向的、一次性的：producer 发完就完，consumer 用完就丢。

### 2.2 架构

```
外部请求 → 全局 Proxy (9000)        ← 用户只跟 Proxy 交互
              │
              ├─→ Prefill 节点 (8100)  ─┐
              │     计算 KV cache       │ P2P 直传 KV cache
              │                         │ (数据面不经过 Proxy)
              └─→ Decode 节点 (8200)  ←─┘
                    接收 KV cache 后 decode
```

**Proxy 的职责：**

Proxy 是 PD 分离的"调度中心"——没有它，prefiller 和 decoder 之间虽然能 P2P 传 KV cache，但**外部请求不知道发给谁**。

| 职责 | 说明 |
|---|---|
| 请求路由 | 接收用户的 OpenAI 兼容请求，把 prefill 阶段转发给 Prefill 节点，decode 阶段转发给 Decoder 节点 |
| 负载均衡 | 支持多对 Prefill/Decoder 实例（`--prefiller-hosts` / `--decoder-hosts`），自动分发请求 |
| 流式透传 | 把 Decoder 的流式输出（SSE）透传给客户端 |
| 健康检查 | 提供 `/healthcheck` 端点，返回连接的 prefiller/decoder 数量 |

**Proxy 的实现：**

E2E 测试框架直接使用仓库中的示例脚本启动 Proxy：

```python
# tests/e2e/nightly/multi_node/internal_dp/scripts/multi_node_config.py:121-124
self.proxy_script = envs.get(
    "DISAGGREGATED_PREFILL_PROXY_SCRIPT",
    "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py",
)
```

`ProxyLauncher` 在 master 节点上自动拉起这个脚本，把 prefiller 和 decoder 的 IP/端口传进去：

```bash
python load_balance_proxy_server_example.py \
    --host <master_ip> --port 9000 \
    --prefiller-hosts <p_ip1> <p_ip2> \
    --prefiller-ports 8100 8100 \
    --decoder-hosts <d_ip1> <d_ip2> \
    --decoder-ports 8200 8200
```

**关键：Proxy 只参与控制面，不参与数据面。** KV cache 的 P2P 传输走 Mooncake RDMA，完全不经过 Proxy——否则 Proxy 会成为瓶颈，PD 分离就没意义了。

### 2.3 两种传输模式

| 模式 | Connector | 工作原理 |
|---|---|---|
| **Pull**（D 拉取） | `MooncakeConnectorV1` | Proxy 路由到 P 完成 prefill → D 节点主动从 P 拉取 KV cache |
| **Push**（P 推送） | `MooncakeLayerwiseConnector` | P 逐层算完一层立即推送给 D → D 逐层接收后开始 decode，延迟更低 |

### 2.4 代码证据

**底层传输引擎：**

```python
# kv_p2p/mooncake_connector.py:25
from mooncake.engine import TransferEngine
```

三个 connector 都依赖 `mooncake.engine.TransferEngine` 做底层 P2P 传输。

**角色分工：**

```python
# kv_p2p/mooncake_connector.py:2327
if self.kv_role == "kv_producer":
    self.kv_send_thread = KVCacheSendingThread(...)  # producer 发送 KV

# kv_p2p/mooncake_connector.py:2382
if self.kv_role == "kv_consumer":
    self.kv_recv_thread = KVCacheRecvingThread(...)  # consumer 接收 KV
```

`kv_producer` 跑在 prefill 节点，`kv_consumer` 跑在 decode 节点。

### 2.5 三种 connector 的差异

| Connector | 传输模式 | 差异 |
|---|---|---|
| `MooncakeConnectorV1` | Pull | 按 request 粒度传输，一批请求算完后一起发 |
| `MooncakeLayerwiseConnector` | Push | 按 layer 粒度逐层传输，算完一层立即发一层，延迟更低 |
| `MooncakeHybridConnector` | Pull | 继承 V1，处理 MLA/Full Attention 混合 block size 的模型（如 DeepSeek-V4） |

### 2.6 一句话总结

> **PD 分离 = prefill 节点算出来 → P2P 直接发给 decode 节点 → decode 节点用。**
> 
> 数据流是单向的、一次性的，不持久化。

---

## 三、池化（kv_pool）：存储 → 查找 → 加载

### 3.1 核心机制

池化的本质是：**KV cache 先存到一个"池子"里，后续请求通过 key 查找命中，再加载回来复用**。数据流是"存→查→取"的循环，同一个 KV block 可能被多个请求命中加载。

### 3.2 两种部署模式

| 部署模式 | kv_role | 说明 |
|---|---|---|
| **PD-Mixed** | `kv_both` | 单实例自己存自己取，池子作为共享 prefix cache |
| **PD 分离 + 池化** | `kv_producer` / `kv_consumer` | P 节点存入池子，D 节点从池子加载，通过 `MultiConnector` 组合 `MooncakeConnectorV1` + `AscendStoreConnector` |

### 3.3 代码证据

**池化专用组件（kv_p2p 中没有）：**

```python
# kv_pool/ascend_store/ascend_store_connector.py:32-36
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,    # 池化调度器
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker         # 池化 worker
)
```

**存储（Save）：**

```python
# kv_pool/ascend_store/ascend_store_connector.py:224-233
def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
    self.connector_worker.save_kv_layer(self._get_connector_metadata())
```

**加载（Load）：**

```python
# kv_pool/ascend_store/ascend_store_connector.py:200-217
def start_load_kv(self, forward_context, **kwargs):
    metadata = self._get_connector_metadata()
    self.connector_worker.start_load_kv(metadata)
```

**查找（Lookup）：**

```python
# kv_pool/ascend_store/pool_scheduler.py:1108
def lookup(self, ...):
    ...  # 按 block hash 查找池中是否已有缓存
```

**外部缓存池：**

```python
# kv_pool/ascend_store/coordinator.py:27
class ExternalCachedBlockPool:
    """Duck-typed BlockPool backed by external AscendStore key existence."""
```

### 3.4 五个池化 connector

| Connector | 后端 | 部署模式 | 外部依赖 |
|---|---|---|---|
| `AscendStoreConnector` | Mooncake Store / Memcache / Yuanrong | `kv_both`、`kv_producer`/`kv_consumer` | Mooncake/Memcache 服务 |
| `SimpleCPUOffloadConnector` | CPU DRAM | `kv_both` | 无 |
| `RecomputeCPUOffloadConnector` | CPU DRAM | `kv_both` | 无 |
| `LMCacheAscendConnector` | LMCache 后端 | `kv_both` | `lmcache_ascend` 库 |
| `UCMConnector` | UCM | `kv_both` | `ucm` 库 + UCM 服务 |

> `AscendStoreConnector` 还支持 **Layerwise 模式**（`use_layerwise: true`），以逐层方式 save/load KV cache，减少首 token 延迟。当前仅支持 Memcache 后端。

### 3.5 一句话总结

> **池化 = 先存到池子里 → 后续请求按 key 查找 → 命中就加载复用。**
> 
> 数据流是"存→查→取"的循环，KV cache 持久化在池中，跨请求复用。

---

## 四、关键区别对照

| 维度 | PD 分离（kv_p2p） | 池化（kv_pool） |
|---|---|---|
| 本质 | KV cache **传输** | KV cache **存储** |
| 目录 | `kv_p2p/` | `kv_pool/` |
| 核心组件 | `TransferEngine`（Mooncake P2P 传输引擎） | `KVPoolScheduler` + `KVPoolWorker` + `ExternalCachedBlockPool` |
| 角色 | `kv_producer`（发） / `kv_consumer`（收） | `kv_both`（既存又取） |
| 数据流 | producer → consumer（单向，一次性） | store → pool → lookup → load（循环，复用） |
| 数据生命周期 | 不持久化，用完即丢 | 持久化在池中（DRAM/SSD），跨请求复用 |
| 关键方法 | `send_kv` / `recv_kv` | `save_kv_layer` / `start_load_kv` / `lookup` |
| Mooncake 作用 | P2P Transfer Engine（RDMA 直传） | Mooncake Store（持久化存储 + master 服务） |
| 可共存 | 可以！通过 `MultiConnector` 同时启用 |

### 4.1 为什么 PD 分离不需要存储，而池化需要？

本质区别在于**消费者是否"在线等着"**。

**PD 分离：消费者在等着**

```
时间轴 ──────────────────────────────────────────→

Prefill 节点:  [算 KV cache] ──RDMA直传──→ 释放
Decode 节点:   [  空转等待  ] ←─收到──→ [decode decode decode...]

P 和 D 是同时存在的，D 等着 P 算完，P 算完立刻传，D 立刻用。
```

- P 和 D **同时在线**，D 正空转等着 P 的 KV cache 来干活
- P 算完直接 RDMA 塞给 D，不需要存——传完就丢，因为 D 已经接住了
- 就像两个人面对面递东西：A 递出来，B 立刻接住，不需要放地上

**池化：没人等着**

```
时间轴 ──────────────────────────────────────────→

请求 A:  [算 KV cache] → 存入池子 → 结束
                                        ...（可能几秒、几分钟后）
请求 B:                              [查找池子] → 命中！→ 加载复用 → 开始 decode

A 和 B 是不同时间的请求，B 来的时候 A 早就结束了。
```

- 当前请求算完 KV cache 时，**没有人在等它**
- 未来请求可能几秒、几分钟后才来，甚至不来
- 所以必须**持久化到池子里**，等未来请求来了再查、再加载
- 就像把东西存到仓库：现在用不上，但以后可能有人来取

**用代码对照：**

| | PD 分离 | 池化 |
|---|---|---|
| 数据流向 | P → D（同时在线，传完即丢） | 存入池子 → ... → 从池子加载（不同时间） |
| 关键操作 | `batch_transfer_sync_read`（RDMA 读远端 NPU 显存） | `save_kv_layer` → `lookup` → `start_load_kv`（put/get 到后端存储） |
| 为什么需要存储 | **不需要**，因为 D 立刻接住就用 | **需要**，因为消费者可能很久以后才来 |

> **一句话：PD 分离是"我在等你"，所以直接递给你就行；池化是"我不在，先放仓库，以后来取"。**

### 4.2 代码印证：PD 分离确实不落盘

从代码可以清晰证明 PD 分离不持久化，四层证据链互相印证：

**证据一：传输引擎初始化 —— 明确声明 P2P 模式，不是存储模式**

```python
# kv_p2p/utils/mooncake_transfer_engine.py:28
self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
```

`"P2PHANDSHAKE"` 是 Mooncake 的 P2P 直传模式，和存储模式（Mooncake Store）是两套完全不同的 API。如果是存储模式，初始化参数会是 `"apollo"` 或 `"memcache"` 这类后端地址。

**证据二：传输 API —— 只有 RDMA 读，没有任何存储 API**

```python
# kv_p2p/mooncake_connector.py:905
ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
```

PD 分离的整个收发流程中，**只有 `batch_transfer_sync_read` 这一个数据传输 API**。代码里没有任何 `put`、`get`、`save`、`load`、`store` 等存储操作。

对比池化，池化的代码里全是存储 API：

```python
# kv_pool/ascend_store/ascend_store_connector.py:224-233
def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
    self.connector_worker.save_kv_layer(...)  # 存入池子

# kv_pool/ascend_store/ascend_store_connector.py:200-217
def start_load_kv(self, forward_context, **kwargs):
    self.connector_worker.start_load_kv(...)   # 从池子加载
```

**证据三：传输完成后立即释放，不持久化**

```python
# kv_p2p/mooncake_connector.py:361-362
elif msg[0] == DONE_RECVING_MSG:
    logger.debug("Got DONE_RECVING_MSG for request %s", msg[1])
```

D 节点收到 KV cache 后，通过 ZMQ 发 `DONE_RECVING_MSG` 给 P 节点。P 节点收到后，把 KV block 放入 free 队列——**释放**，不是**存储**：

```python
# kv_p2p/mooncake_connector.py:168-175
class KVCacheTaskTracker:
    # Only used in prefill node. Tracks requests whose kv blocks freeing is
    # intentionally delayed. Each entry is a tuple of (request_id, timestamp).
    # If a request remains in this queue for too long, it will be force-freed.
    self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
```

```python
# kv_p2p/mooncake_connector.py:191
self.delayed_free_requests.pop(request_id, None)  # 正常释放

# kv_p2p/mooncake_connector.py:233
"Force freed expired request: %s. "  # 超时强释放
```

如果 KV cache 是落盘持久化的，就不会有"force free"这种逻辑——持久化意味着想留多久留多久，不会过期强释放。

**证据四：PD 分离代码里完全没有池化组件**

搜遍整个 `kv_p2p/` 目录：

| 池化核心组件 | `kv_p2p/` 搜索结果 |
|---|---|
| `KVPoolScheduler` | 0 个结果 |
| `KVPoolWorker` | 0 个结果 |
| `ExternalCachedBlockPool` | 0 个结果 |
| `save_kv_layer` | 0 个结果 |
| `start_load_kv` | 0 个结果 |
| `lookup` | 0 个结果 |

这些是池化的核心组件，PD 分离一个都没有。如果 PD 分离需要落盘，至少要有类似 `save_kv_layer` 的调用。

**四层证据总结：**

| 证据层 | 结论 |
|---|---|
| 初始化参数 | `"P2PHANDSHAKE"` — 明确 P2P 模式，不是存储模式 |
| 传输 API | 只有 `batch_transfer_sync_read`（RDMA），没有 `put`/`get`/`save`/`load` |
| 传输后行为 | `DONE_RECVING` → `delayed_free_requests` → 释放 block，不是存储 |
| 组件缺失 | `kv_p2p/` 中没有任何池化组件 |

> **代码里没有一行把 KV cache 写到磁盘或任何后端存储，全在 NPU 显存之间 RDMA 直传，传完就释放。**

---

## 五、常见误解

1. **"PD 分离和池化是一回事"** —— 错。一个在 `kv_p2p/`，一个在 `kv_pool/`，是两套独立代码，本质不同（传输 vs 存储）。

2. **"PD 分离是池化的一种"** —— 错。PD 分离是直传（不持久化），池化是存储复用（持久化），数据流完全不同。

3. **"池化不能跨节点"** —— 错。`AscendStoreConnector` 通过 Mooncake Store 后端，可以跨节点存取 KV cache。PD 分离 + 池化部署模式下，P 节点存入池子，D 节点从池子加载。

4. **"池化需要 PD 分离"** —— 错。`AscendStoreConnector` 在 PD-Mixed 模式（`kv_both`）下单节点独立运行，不需要 PD 分离。

5. **"`pull_request/one_card/pooling/*` 是 KV cache 池化"** —— 错。那是 embedding/classification 的 pooling runner（表示层池化，对向量做平均/最大池化），与 KV cache 池化无关。

6. **"多节点就是 PD 分离"** —— 错。多节点 YAML 可能是纯分布式部署（如 `DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml`），不含 `disaggregated_prefill` 字段。

---

## 六、附录：Mooncake 如何做到 "不存储，直接传输"？

PD 分离中，Mooncake 的 P2P 直传基于 **RDMA 单边读**，不经过任何中间存储：

```
P 节点 (NPU 显存)                          D 节点 (NPU 显存)
     │                                          │
     │ ① initialize("P2PHANDSHAKE", "ascend")   │
     │    register_memory(ptr, size)             │
     │    把 NPU 显存地址注册到 TransferEngine   │
     │                                          │
     │ ② 通过 ZMQ side channel 发送：            │
     │    - session_id（远端内存句柄）             │
     │    - block 内存布局信息                    │
     │                                          │
     │                                   ③ batch_transfer_sync_read(
     │                                          session_id,     ← 远端内存句柄
     │                                          src_ptrs,       ← 远端 NPU 地址
     │                                          dst_ptrs,       ← 本地 NPU 地址
     │                                          lengths         ← 数据长度
     │                                      )
     │                                          │
     │     ╔════════ RDMA 单边读 ═══════════╗     │
     │     ║ P 节点 NPU 显存 ──────────→ D 节点 NPU 显存
     │     ║ 不经过 CPU，不经过磁盘       ║
     │     ╚══════════════════════════════╝
     │                                          │
     │ ④ ZMQ 通知 DONE_RECVING                   │
     │    → P 释放 KV cache                      │
```

**关键代码：**

```python
# mooncake_transfer_engine.py — 初始化传输引擎
self.transfer_engine = TransferEngine()
self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
self.transfer_engine.register_memory(ptr, size)  # 注册 NPU 显存，暴露给远端

# mooncake_connector.py — D 节点直接读 P 节点的 NPU 显存
session_id = f"{remote_host}:{remote_transfer_port}"
self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
```

**核心要点：**

- `register_memory(ptr, size)` 把本地 NPU 显存地址注册到 Mooncake TransferEngine，使其可被远端访问
- `session_id` 是远端内存的句柄（`host:port` 格式），携带它就能访问远端已注册的内存
- `batch_transfer_sync_read` 是 **RDMA 单边读**（one-sided RDMA read）——D 节点像读自己本地内存一样直接读 P 节点的 NPU 显存
- 全程数据不落盘、不经过 CPU，从 NPU 到 NPU 直通
- 传输完成后，P 节点释放 KV cache，不持久化任何数据

这就是为什么叫 "不存储，直接传输"——Mooncake 的 TransferEngine 提供的是对远端 NPU 显存的直接 RDMA 访问能力，数据从一块 NPU 显存直通到另一块 NPU 显存，不存在 "先存到某个地方再取出来" 的中间环节。