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
**`c128` 可以近似理解为“这个 KV cache group 的序列长度被压缩了 128 倍”，即原始 128 个 token 范围对应 1 个压缩 KV 位置；但这个压缩 KV 是模型网络通过训练好的参数计算出来的，不是推理框架或 KV Pool 临时把 128 个 token 简单压成 1 个 token。**