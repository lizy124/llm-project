# KV 连接器架构问题汇总

## Q1: 为什么需要 KV 连接器注册机制？

**简要回答**：
vLLM 是通用推理框架，需支持多种硬件平台。上游只定义接口规范，各厂商实现自己的连接器，通过注册机制"告诉"上游有哪些实现可用。实现"接口在上游，实现在下游"的分层架构。

---

## Q2: `register_connector(name, module_path, class_name)` 三个参数的含义？

**简要回答**：
- `name`：配置名称，用户在 YAML 中使用的字符串
- `module_path`：Python 模块路径（延迟加载）
- `class_name`：模块中的实际类名

注册时只记录路径，真正 import 发生在使用时，实现延迟加载。

---

## Q3: 为什么一个类（AscendStoreConnector）同时用于 Scheduler 和 Worker？

**简要回答**：
vLLM 的 Scheduler 进程和 Worker 进程读取**同一个配置文件**，只能配置一个连接器名称。一个类通过 `role` 参数内部分支：
- `role=SCHEDULER` → 创建 `KVPoolScheduler`
- `role=WORKER` → 创建 `KVPoolWorker`

这样同一个配置名称，两个进程都能正常工作。

---

## Q4: 既然有 KVPoolScheduler 和 KVPoolWorker，为什么还需要 AscendStoreConnector？

**简要回答**：
AscendStoreConnector 是**适配器/包装器**：
1. 满足 vLLM 接口规范（必须同时实现 Scheduler 和 Worker 方法）
2. 适配工厂模式（一个类名同时支持两种进程）
3. 组合内部组件（将 Scheduler 和 Worker 逻辑分离）
4. 代理方法调用（根据 role 决定代理到哪个内部组件）

---

## Q5: 为什么 vLLM 要求一个接口同时实现 Scheduler 和 Worker 方法？

**简要回答**：
简化用户配置和工厂设计：
- 用户只需配置一个名称，不用分别配置 Scheduler 和 Worker
- 工厂只需一个注册表，不用两个
- 避免配置匹配问题（如 Scheduler 和 Worker 配置不兼容）

设计理念：**用户友好 + 代码简洁** > **严格的接口分离**

---

## Q6: Scheduler 端的调度是池化调度吗？它和 vLLM 调度是什么关系？

**简要回答**：
是的，文档中说的 Scheduler 端调度主要是 **KV Pool / 外部 KV Cache 池化相关的调度辅助逻辑**，不是替代 vLLM 原生 Scheduler 的完整请求调度。

更准确地说：**vLLM Scheduler 是主调度器，KVPoolScheduler 是嵌入 vLLM Scheduler 调度流程中的 KV Cache 池化协同模块**。

二者关系：
- **vLLM Scheduler**：负责整体推理调度，例如哪些请求本轮运行、prefill/decode 如何安排、token budget 如何分配、本地 KV block 如何分配、请求是否抢占等。
- **KVPoolScheduler**：只负责外部 KV Pool 相关问题，例如请求在外部池中命中多少 KV Cache、命中后需要预留多少本地 block、哪些请求或 chunk 的 KV Cache 需要保存到外部池，以及构造 Worker 端 load/save 所需元数据。

典型调用关系：
```text
vLLM Scheduler 调度循环
    ↓
KVPoolScheduler.get_num_new_matched_tokens()
    ↓
vLLM Scheduler 根据 hit 数量决定分配多少 block
    ↓
KVPoolScheduler.update_state_after_alloc()
    ↓
KVPoolScheduler.build_connector_meta()
    ↓
Worker 端执行 load/save
```

一句话总结：
**池化调度不是另一个完整的 vLLM 调度器，而是嵌入 vLLM Scheduler 流程中的“KV Cache 复用 / 加载 / 保存决策层”。它影响 vLLM 的 block 分配和本轮需要计算的 token 数，但不接管 vLLM 的整体请求调度策略。**

---

## Q7: hybrid KV cache 是什么？它是一种 KV cache 吗？

**简要回答**：
`hybrid KV cache` 不是一种单独的新 KV cache 数据结构，而是一种 **KV cache 管理形态**。

简单说：
- **普通 KV cache**：所有层 / 所有 attention 使用同一种 KV cache 规格，通常都是 `FullAttentionSpec`。
- **hybrid KV cache**：同一个模型里存在多种不同规格的 KV cache，需要按 group 分组管理。

例如普通 Transformer 模型中，所有层可能都是 full attention：
```text
Layer 0  Full Attention  → KV cache
Layer 1  Full Attention  → KV cache
Layer 2  Full Attention  → KV cache
...
```

这种情况下 KV cache 的形状、block 规则、可复用范围基本一致，可以按统一规则管理。

而 hybrid KV cache 常见于混合注意力或特殊 KV 规格模型，例如：
```text
Layer 0  Full Attention       → KV cache group A
Layer 1  Sliding Window Attn  → KV cache group B
Layer 2  Full Attention       → KV cache group A
Layer 3  Sliding Window Attn  → KV cache group B
...
```

或者类似 DeepSeek 这类存在不同 KV 压缩 / MLA 规格的情况：
```text
KV group 0: c1    cache
KV group 1: c4    cache
KV group 2: c128  cache
```

这些 group 的 KV cache 可能在以下方面不同：
- attention 类型不同；
- 每个 token 对应的 KV 数据大小不同；
- block size 不同；
- cache shape 不同；
- 可缓存 token 范围不同；
- 保存 / 加载粒度不同。

因此需要 `hybrid KV cache manager` 按 group 分别管理。

结合代码：
```python
return len(kv_cache_config.kv_cache_groups) > 1 and any(
    not isinstance(group.kv_cache_spec, FullAttentionSpec)
    for group in kv_cache_config.kv_cache_groups
)
```

它判断的是：
1. 是否存在多个 KV cache group；
2. 是否至少有一个 group 不是普通 `FullAttentionSpec`。

只有同时满足这两个条件，才认为当前使用 hybrid KV cache。

一句话总结：
**hybrid KV cache 不是一种具体 cache，而是“同一个模型里混用了多种 KV cache 规格，因此需要分组管理”的模式。**

---

## Q8: DeepSeek V4 里的 c4 / c128 到底是什么意思？c128 是把 128 个 token 的特征压缩成 1 个 token 特征吗？

**简要回答**：
`c4` / `c128` 可以理解为某些 KV cache group 在**序列长度维度上的压缩倍率 / 稀疏倍率**。其中 `c128` 更接近表示：这个 KV cache group 的 cache 长度大约是原始 token 长度的 `1/128`。

也就是说：
```text
普通 c1 KV：   每 1 个 token 对应 1 个 KV 位置
c4 KV：        大约每 4 个 token 对应 1 个 KV 位置
c128 KV：      大约每 128 个 token 对应 1 个 KV 位置
```

但要注意，`c128` **不是**简单地把 128 个 token 的 hidden state 做平均池化，压缩成一个普通 token 特征。

更准确地说：
> `c128` 对应的是模型网络结构内部某个压缩 attention / latent KV 分支产生的 KV 表示。这个 KV 表示是通过训练好的网络参数计算出来的，不是 KV Pool 或推理框架临时压缩出来的。

可以这样理解普通 KV 和 c128 KV 的区别：

```text
普通 full KV / c1：
tokens:  t1   t2   t3   t4   ...   t16384
KV:      kv1  kv2  kv3  kv4  ...   kv16384

c128 KV：
tokens:  t1 ... t128 | t129 ... t256 | ...
KV:           kv_1   |      kv_2     | ...
```

这里的 `kv_1` 不是某一个 token 的 hidden state，也不是普通意义上的“一个 token 特征”。它通常是模型结构中某个专门模块根据一段 token 信息计算出来的压缩 KV 表示。

从模型角度看：

```text
输入 token hidden states
        ↓
普通 attention / KV 分支       → 产生 c1 KV
压缩 attention / KV 分支       → 产生 c4 / c128 KV
MLA / latent KV 分支           → 产生特殊压缩 KV
```

这些分支中的投影矩阵、压缩方式、latent 表示方式，都是模型结构的一部分，并且在训练阶段已经参与训练。

所以训练阶段会学习：
- 如何把一段 token 范围的信息表达成较少的 KV 表示；
- 后续 token 如何利用这些压缩 KV 做 attention；
- 压缩后丢失的信息如何通过模型结构和参数补偿。

推理阶段不会重新设计压缩方式，只是执行训练好的网络：

```text
模型网络决定：c128 KV 怎么算
训练过程学习：c128 KV 里应该保留什么信息
推理过程执行：用训练好的权重算出 c128 KV
KV Cache 负责：把算出来的 c128 KV 缓存起来
KV Pool 负责：按正确粒度保存、加载和复用这些 KV
```

因此，`c128` 对 KV Pool / Scheduler 的核心影响不是“怎么压缩”，而是“怎么对齐和传输”。

例如 block size 是 128 tokens 时：
```text
c1:    1 个 cache block 对应 1   × 128 = 128 tokens
c4:    1 个 cache block 对应 4   × 128 = 512 tokens
c128:  1 个 cache block 对应 128 × 128 = 16384 tokens
```

所以文档里才会说 DeepSeek V4 的 KV Cache 存取最终需要按 `16384 tokens` 这样的粒度对齐。

一句话总结：
**`c128` 可以近似理解为"这个 KV cache group 的序列长度被压缩了 128 倍"，即原始 128 个 token 范围对应 1 个压缩 KV 位置；但这个压缩 KV 是模型网络通过训练好的参数计算出来的，不是推理框架或 KV Pool 临时把 128 个 token 简单压成 1 个 token。**

---

## Q9: vllm-ascend 仓库涉及到的图模式是什么？

**简要回答**：
vLLM Ascend 的图模式有两层含义：

**一、NPU 图捕获/回放（ACLGraph）**
核心文件：`vllm_ascend/compilation/acl_graph.py`

对标 NVIDIA 的 CUDA Graph，原理是：Capture 阶段记录一连串 NPU 算子操作 → Replay 阶段一次性回放，省去逐算子调度的开销。

支持三种模式（通过 `CUDAGraphMode` 枚举控制）：

| 模式 | 说明 |
|------|------|
| `NONE` | 不用图，逐算子执行 |
| `FULL` | 整模型捕获为一个图，decode 阶段回放 |
| `FULL_DECODE_ONLY` | 仅 decode 阶段使用图 |
| `PIECEWISE` | 按层/分段捕获，每段独立回放（Ascend 默认禁用） |

在 `vllm_ascend/platform.py` 中根据 Ascend 特性做兼容处理，默认禁用 PIECEWISE 模式。

**二、FX 图融合编译（Graph Fusion）**
核心文件：`vllm_ascend/compilation/graph_fusion_pass_manager.py`

在 PyTorch FX 图层面做算子融合优化，包括：
- Norm + Quant 融合
- QKNorm + Rope 融合
- AllReduce + RMSNorm 融合
- 序列并行优化
- Noop 消除

**为什么 ACLGraph 里没法打日志？**
这是图捕获机制本身的固有限制：
1. Capture 阶段：NPU 只记录算子操作，不真正执行计算，Python 代码照常运行，可以打日志
2. Replay 阶段：整张图作为一个整体提交给 NPU 回放，中间不经过 Python，没有任何 Python 回调机会，自然打不了日志

不是代码没写，而是图回放期间 Python 根本没有介入点。

---

## Q10: `self.client` 是什么？`lookup` 的用法是什么？

**简要回答**：
`self.client` 是 `LookupKeyClient` 的实例，本质是一个基于 **ZMQ REQ/REP** 的 RPC 客户端。

```python
def lookup(
    self,
    token_len: int,                          # 当前请求的 token 长度
    block_hashes: list[BlockHash],            # 每个 block 的哈希值列表
    kv_cache_group_ids: list[int] | None = None,  # KV cache 分组 ID
) -> int:  # 返回外部命中的 token 数量
```

**做了什么**：把 block 哈希通过 ZMQ 发给 Worker 进程，Worker 在远端 KV cache 中查找有哪些 block 已经缓存了（命中），返回命中的 token 数量。Scheduler 据此决定哪些 block 可以直接复用。

---

## Q11: `LookupKeyClient` 是和 Mooncake 的 master 通信吗？

**简要回答**：
是的，完整链路如下：

```
LookupKeyClient.lookup()          ← Scheduler 进程
        │
        │ ZMQ IPC (REQ/REP)
        ▼
LookupKeyServer                   ← Worker 进程 (rank 0)
        │
        │ 调用 pool_worker.lookup_scheduler()
        ▼
self.m_store.exists(multi_tp_keys)  ← Mooncake 后端
```

关键证据：
1. `self.m_store` 默认就是 Mooncake 后端（`pool_worker.py:L107`）：`self.backend = vllm_config.kv_transfer_config.kv_connector_extra_config.get("backend", "mooncake")`
2. `lookup_scheduler` 最终调用 `self.m_store.exists(multi_tp_keys)`（`pool_worker.py:L1024`）
3. ZMQ 路径中也有 `mooncake_rpc_port` 的配置兼容

`LookupKeyClient` 通过 IPC 与本节点 rank 0 的 Worker 进程通信，Worker 最终调用 Mooncake 分布式 KV Store 的 `exists()` 接口。

---

## Q12: 完全命中时为什么还要减 1？

**简要回答**：
```python
if num_external_hit_tokens == request.num_tokens:
    num_external_hit_tokens -= 1
```

这是 vLLM 前缀缓存的标准约定：**不能让全部 token 都命中，必须留至少 1 个 token 给本地计算**。

vLLM 的执行流程是：算完 prompt → 产生第一个 output token。如果所有 token 的 KV cache 都从外部命中，模型就不需要做任何 forward 计算，会导致：
1. 没有 forward pass，无法产生第一个输出 token
2. 执行管线断掉，因为 vLLM 调度器期望至少执行一次模型推理

减 1 让前 `num_tokens - 1` 个 token 走外部 KV cache 复用，最后一个 token 必须本地计算，保证推理管线正常衔接。

---

## Q13: 所有 KV cache 都命中，为什么不能继续 decode？

**简要回答**：
区分两个概念：**KV cache 命中 ≠ 不需要 forward**。

KV cache 存的是每一层的 K 和 V。但模型要预测下一个 token，需要的是：
```
Q_last_token × K_all_cached → attention → logits → next_token
```

即使所有 K、V 都缓存在外部，**最后一个 token 的 Q 向量也必须通过模型 forward 算出来**。没有 Q，就没法做 attention，也就没法预测下一个 token。

减 1 的作用：告诉调度器"别全跳过，最后一个 token 跑一次 forward"。这次 forward 做的事是：第 N-1 个 token 的 Q 向量 × 前 N-1 个 token 的 K、V（外部已缓存）→ 产生第 N 个 token（第一个输出）。

---

## Q14: `kv_cache_group_ids` 是什么？和 `block_hashes` 一一对应吗？

**简要回答**：
**不是一一对应。**

`kv_cache_group_ids` 是 Hybrid 模型的 KV cache 分组 ID。比如一个模型同时有 Full Attention 层和 Sliding Window Attention 层，它们属于不同的 KV cache group，每个 group 有独立的 block 池。

初始化时：
```python
self.kv_cache_group_ids = (
    list(range(len(kv_cache_config.kv_cache_groups)))  # 有多个 group
    if kv_cache_config is not None and self.use_hybrid
    else [0]  # 普通模型只有一个 group
)
```

调用时 `kv_cache_group_ids` 是所有 group 的集合（如 `[0, 1, 2]`），`block_hashes` 是当前请求的 block 哈希列表。服务端会对**每个 group** 都遍历一遍 `block_hashes`：

```python
for group_id in kv_cache_group_ids:
    for start, end, key in self.token_database.process_tokens(
        token_len, block_hashes, kv_cache_group_id=group_id,
    ):
```

| 概念 | 含义 |
|------|------|
| `kv_cache_group_ids` | 所有需要查询的 KV cache 分组 ID |
| `block_hashes` | 某个请求的 block 哈希列表 |
| 关系 | 每次 lookup，所有 group 都用同一组 block_hashes 查一遍 |

---

## Q15: `all_frames` 的长度是固定的吗？

**简要回答**：
**不固定。** 长度取决于 `block_hashes` 的数量。

帧结构：
```
all_frames[0]     → token_len_bytes        (固定 1 帧)
all_frames[1]     → kv_group_frames        (固定 1 帧，msgpack 编码的 group 列表)
all_frames[2:]    → hash_frames            (变长，msgpack 编码的 block 哈希列表)
```

服务端解码时也印证了这一点：
```python
token_len = int.from_bytes(all_frames[0], byteorder="big")   # 固定第 0 帧
kv_group_ids = self.decoder.decode([all_frames[1]])           # 固定第 1 帧
hash_frames = all_frames[2:]                                  # 剩下全吃掉
```

`hash_strs` 是 `block_hashes` 的 hex 字符串列表，每个请求的 block 数量不同 → 列表长度不同 → msgpack 编码后产生的 frame 数量也不同。总长度 = `2 + N`，N 随请求变化。

---

## Q16: `send_multipart` 是发给谁了？

**简要回答**：
发给 **`LookupKeyServer`**，它运行在本节点 rank 0 的 Worker 进程中。

```
LookupKeyClient (Scheduler 进程)           LookupKeyServer (Worker 进程, rank 0)
         │                                           │
         │  ZMQ REQ  ────ipc://.../lookup_rpc_port──▶  ZMQ REP
         │              (bind=False, connect)        (bind=True, listen)
         │                                           │
         │  ◀─────────────── response ───────────────│
```

两端用的是同一个 IPC 路径，由 `get_zmq_rpc_path_lookup()` 生成。每个节点只有一个 Worker 进程（rank 0）持有 Mooncake 的 `m_store` 连接，所以所有 lookup 请求都发给它。

---

## Q17: 为什么 `LookupKeyServer` 在 `AscendStoreConnector` 里初始化，而不是在 `KVPoolWorker`？

**简要回答**：
这是**职责分离**的设计：

| 类 | 职责 |
|----|------|
| `KVPoolWorker` | 纯业务逻辑：KV 池的 save/load/lookup，调用 Mooncake store |
| `LookupKeyServer` | ZMQ 传输层：收包、解包、调 worker、回包 |
| `AscendStoreConnector` | 编排者：根据 role 创建 Scheduler 或 Worker，并决定是否开启 ZMQ 服务 |

`KVPoolWorker` 不需要知道自己是被 ZMQ 调用还是进程内直接调用——它只是一个纯逻辑组件。如果把它塞进 `KVPoolWorker`，就绑死了 ZMQ 通信方式，以后换 gRPC、共享内存等就得改 worker 代码。

`AscendStoreConnector` 作为 vLLM 框架的入口点，天然知道 `role`（Scheduler/Worker）和 `rank`，在这里做组件编排最合适。

---

## Q18: Scheduler 和 Worker 之间的通信，除了池化查询，还有其他通信吗？都用 ZMQ 吗？

**简要回答**：
只有一条 ZMQ 通道。Scheduler 和 Worker 之间的通信一共有三层：

| 层级 | 机制 | 方向 | 内容 |
|------|------|------|------|
| ① 池化查询 | ZMQ REQ/REP | Scheduler → Worker | 查 block 哈希是否命中 |
| ② 元数据传递 | vLLM 框架内置 | Scheduler ↔ Worker | 告诉 Worker 该 load/save 哪些 block |
| ③ 数据传输 | Mooncake 协议 | 各 Worker 之间 | KV cache 的实际读写 |

第②层：vLLM 的 `KVConnectorBase_V1` 框架自带元数据通道。`build_connector_meta()` 返回 `KVConnectorMetadata`，框架自动传给 Worker；Worker 通过 `update_connector_output()` 把 KV events 回传给 Scheduler。

第③层：KV cache 的实际存/取走 Mooncake 分布式存储协议，Worker 直接和 Mooncake store 交互，不经过 Scheduler。

ZMQ 只用于那一条查询链路，因为查询是**同步的**（Scheduler 必须等结果才能决定分配多少 block），其他通信要么走框架自带通道，要么走 Mooncake 自己的协议。

---

## Q19: vLLM 本身的 Scheduler-Worker 通信还有哪些？

**简要回答**：
vLLM V1 中 Scheduler 和 Worker 的通信远超池化功能，主要分三类：

**一、核心调度通信（vLLM 框架内置，非 ZMQ）**

| 通信内容 | 方向 | 说明 |
|---------|------|------|
| SchedulerOutput | Scheduler → Worker | 调度结果：哪些请求、多少 token、block 分配 |
| ModelRunnerOutput | Worker → Scheduler | 推理结果：logits、采样 token |
| GrammarOutput | Scheduler → Worker | 结构化输出约束（grammar/JSON schema） |
| IntermediateTensors | PP rank 之间 | 流水线并行中间张量传递 |

这些通过 vLLM 的 Executor 框架（multiprocessing 或 Ray）传递，不走 ZMQ。

**二、KV Transfer 通信（KV Connector 框架）**

| 通信内容 | 方向 | 说明 |
|---------|------|------|
| KVConnectorMetadata | Scheduler → Worker | 告诉 Worker 哪些 block 需要 load/save |
| KVConnectorOutput | Worker → Scheduler | KV 传输结果、完成事件 |
| Handshake Metadata | Worker ↔ Worker | 初始握手，交换各 worker 能力信息 |

这些嵌入在 `SchedulerOutput` / `ModelRunnerOutput` 中一起传递。

**三、分布式并行通信（HCCL）**

| 通信内容 | 范围 | 说明 |
|---------|------|------|
| All-Reduce | TP group | 张量并行的梯度/激活同步 |
| All-Gather | TP/CP group | 上下文并行、序列并行的 KV 同步 |
| Send/Recv | PP group | 流水线并行的激活传递 |

**总结：只有 lookup 查询走了自定义 ZMQ，其他全是 vLLM 框架自带通道。**

---

## Q20: Request 的核心属性有哪些？`num_tokens` 和 `num_computed_tokens` 的区别？

**简要回答**：

### 核心属性

| 属性 | 类型 | 含义 |
|------|------|------|
| `num_tokens` | property → `len(_all_token_ids)` | 请求当前总 token 数（prompt + output），自动增长 |
| `num_computed_tokens` | 普通字段 | 已计算过 KV cache 的 token 数，调度器手动更新 |
| `num_prompt_tokens` | 字段 | prompt 的 token 数 |
| `num_tokens_with_spec` | property | 含投机解码占位 token 的总数 |
| `num_output_placeholders` | 字段 | 投机解码占位 token 数 |
| `prompt_token_ids` | 列表 | prompt token ID 列表 |
| `_all_token_ids` | 列表 | 所有 token（prompt + output），`num_tokens` 的数据源 |
| `_output_token_ids` | 列表 | 已生成 output token ID 列表 |
| `spec_token_ids` | 列表 | 投机解码草稿 token ID 列表 |
| `block_ids` | 列表 | 已分配 KV cache block ID |
| `block_hashes` | 列表 | 每个 block 的哈希值（前缀缓存查找用） |
| `status` | 枚举 | WAITING / RUNNING / PREEMPTED / FINISHED 等 |
| `max_tokens` | 字段 | 最大生成 token 数 |

### `num_tokens` vs `num_computed_tokens`：为什么需要两个？

**核心区别**：一轮循环中，两个值的更新时机不同。

```
一轮循环的流程：

① schedule:   num_computed_tokens += num_new_tokens   ← 先追上 num_tokens
② forward:    模型计算
③ update:     _all_token_ids.append(新token)          ← num_tokens 又涨了

轮次    num_tokens    num_computed_tokens
──────────────────────────────────────────
第 N 轮开始           101           100    ← 差 1
  ① schedule 后      101           101    ← 追平!
  ③ update 后        102           101    ← 又落后了

第 N+1 轮开始         102           101    ← 差 1
  ① schedule 后      102           102    ← 又追平!
```

`num_tokens` 是 property（自动 = `len(_all_token_ids)`），output 出来就涨；
`num_computed_tokens` 是普通字段，调度时手动赋值。

**永远差一个身位**：`num_tokens` 先跑，`num_computed_tokens` 下一轮调度再追。差值 = 本轮要算的 token 数。

---

## Q21: `_model_forward()` 之后是不是会把 KV cache 保存到池化池子里？

**简要回答**：
不一定，触发条件不是 `is_pooling_model`，而是是否启用了 **KV Connector / KV Transfer**。

在 `D:\lzy\project\kv_pool\code\vllm-ascend\vllm_ascend\worker\model_runner_v1.py` 中，`_model_forward()` 被包在：

```python
with self.maybe_get_kv_connector_output(scheduler_output, defer_finalize=...) as kv_connector_output:
    hidden_states = self._model_forward(...)
```

如果存在 `has_kv_transfer_group()`，这个 context 会绑定 Scheduler 传来的 KV connector metadata，并在 forward 期间配合 attention/KV connector 执行 KV load/save/transfer。退出 context 时会 `wait_for_save()`，并生成 `KVConnectorOutput` 回传给 Scheduler。

`is_pooling_model` 分支只是后处理：

```python
output = self._pool(hidden_states, ..., kv_connector_output)
output.kv_connector_output = kv_connector_output
```

`_pool()` 只根据 `hidden_states` 计算 pooling output，不负责保存 KV cache。

一句话总结：
**如果说的是 embedding/pooling 模型的 pooling，不会因为开启 pooling 就保存 KV；如果说的是 KV Pool / KV Transfer，保存动作由 KV connector context 控制，和 `is_pooling_model` 不是同一个概念。**