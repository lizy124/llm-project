# kv_offload 如何在 NPU 和 CPU 之间搬运 KV？

源码基线：

- vLLM Ascend：`d85e6714a09bef4d9de6b8c05e9425183d46ba23`
- vLLM：`58d3918e3ea0a544ffedadad2ba84559e9c51d8f`

源码位置：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/simple/`
- `code/vllm/vllm/v1/kv_offload/`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`

本文的问题是：`kv_offload` 和 `ascend_store` 有什么本质区别？native 与 simple 两条 offload 路径如何注册 NPU KV cache、构造 CPU block view、发起异步 copy，并把完成状态交还给 vLLM？

---

## 0. 本文要回答的问题

```text
1. kv_offload 保存的是什么，和外部 KV Store 的 key/object 有什么区别？
2. native offload 为什么要适配 vLLM 的 CPU offloading worker？
3. NPU KV tensor 如何转换为可搬运的 block view？
4. simple offload 的 copy backend 和 worker 如何协作？
5. pinned CPU memory、descriptor buffer、DMA thread 分别解决什么问题？
6. load/save 的完成状态如何回到 vLLM connector？
```

---

## 1. 一句话回答

`kv_offload` 是本地 KV tensor 的存储层级搬运：它把 NPU 上的 KV block 复制到 CPU buffer，需要时再复制回 NPU；它不通过 key 查询外部对象，也不负责跨节点 backend 的 lease、alloc 或一致性。

两条实现路径可以先这样区分：

```text
native：复用 vLLM 原生 offloading 抽象，Ascend 负责 layout 和 NPU copy 适配；
simple：复用 vLLM simple offload 生命周期，Ascend 自己替换 worker 和 DMA copy backend。
```

---

## 2. kv_offload 和 ascend_store 的边界是什么？

### 2.1 kv_offload 的对象

offload 的核心对象是本地 tensor/block view：

```text
NPU KV cache tensor
  <-> CPU offload buffer
```

CPU buffer 一般按照 block/page 的大小组织，和本地 GPU/NPU KV cache 的 block layout 对齐。load/save 的定位来自 block id 和 buffer index，而不是外部 hash key。

### 2.2 ascend_store 的对象

`ascend_store` 需要把 token/block hash 变成外部 key，再通过 backend 查询、申请和读写对象。它要处理：

```text
key、lease、exists、alloc、put/get、远端完成状态
```

offload 不具备这些语义。它的主要正确性问题是 tensor layout、地址、copy 完成和 buffer 生命周期。

### 2.3 对比主链

```text
ascend_store:
Scheduler 命中查询 -> metadata -> backend key/address -> 远端数据 -> NPU KV tensor

kv_offload:
vLLM block 分配 -> CPU buffer 索引 -> NPU/CPU copy -> 本地 KV tensor
```

因此 offload 可以独立于外部 KV Store 使用，不能把它当作 `ascend_store` 的简化 backend。

---

## 3. native 路径如何接入 vLLM？

### 3.1 适配关系

`kv_offload/native/offloading_connector.py` 明确复用了 vLLM 的 native offloading connector、配置和 worker 抽象；Ascend 侧主要提供：

```text
AscendOffloadingConnector
AscendOffloadingConnectorWorker
NPUOffloadingSpec
NPUOffloadingWorker
```

`native/npu.py` 中的 `NPUOffloadingSpec` 继承 vLLM CPU offloading spec，并创建 Ascend worker。这样 Scheduler 侧的 offload metadata 和生命周期继续沿用 vLLM 原生实现，Ascend 只替换设备相关部分。

### 3.2 为什么要做 layout adaptation？

vLLM native offloading worker 期望一种规范的 block-level 表示；Ascend 的 KV cache 可能存在：

```text
- K/V 分离或合并；
- attention cache 与其他 state 的不同维度；
- 非连续 stride；
- 每个 layer 的 tensor 形状差异。
```

`AscendOffloadingConnectorWorker.register_kv_caches()` 会把实际 NPU cache 转换成 worker 能识别的 canonical block view。其重点是建立 zero-copy view 或合适的切分 view，而不是重新复制整块 cache。

### 3.3 native copy handler

`native/cpu_npu.py` 中的 `SingleDirectionNPUOffloadingHandler` 批量收集源地址、目标地址和 copy size，创建 descriptor buffer，通过异步传输提交多个 copy operation。

它维护：

```text
- descriptor buffer pool；
- job id 与 TransferResult；
- 已完成和待等待的任务；
- wait/shutdown 生命周期。
```

`NPUOffloadingWorker` 在 vLLM CPU offloading worker 的基础上使用该 handler，把通用 offload 调度转换成 Ascend 批量 DMA 操作。

---

## 4. simple 路径如何工作？

### 4.1 connector 和 worker

`simple/simple_cpu_offload_connector.py` 中的 `AscendSimpleCPUOffloadConnector` 继承 vLLM 的 simple CPU offload connector。它不在构造阶段直接分配所有设备资源，真正的 cache registration 延迟到 Worker 收到实际 KV cache 后执行。

`simple/worker.py` 中的 `SimpleCPUOffloadNPUWorker` 负责：

```text
- 注册 NPU KV caches；
- 创建 CPU block storage 和 block view；
- 维护 load/save 的 block id；
- 调用 NPU copy backend；
- 汇总 finished 状态。
```

### 4.2 block view 的建立

`register_kv_caches()` 会根据 NPU cache tensor 推导每个 block 的地址布局，再构造 CPU storage 对应的 view。这里要求：

```text
NPU view 的 block 数、每 block 字节数、K/V 子 tensor 数量
与 CPU storage 的描述完全一致。
```

如果 cache 是分裂 attention layout，worker 会按 logical state 建立多个 view，而不是假设一个连续 tensor 覆盖所有 K/V。

### 4.3 copy backend

`simple/copy_backend.py` 的 `NPUDmaCopyBackend` 在独立 worker thread 中运行：

```text
launch_copy(src_blocks, dst_blocks, ...)
  -> copy queue
  -> _copy_loop()
  -> npu_mem_ops.copy_blocks()
  -> 完成事件/结果
```

它把 vLLM simple offload worker 原本面向 CUDA 的 copy backend 换成 Ascend NPU 的批量 DMA 实现。

---

## 5. NPU 内存操作如何批量化？

### 5.1 descriptor 参数

`simple/npu_mem_ops.py` 的 `BatchMemcpyParams` 保存一次批量 copy 所需的静态描述，例如：

```text
- 每个子 tensor 的字节数；
- 每个 block 的 stride；
- K/V 或多状态 tensor 的数量；
- block payload 的布局信息。
```

`copy_blocks()` 根据源 block、目标 block 和这些参数展开地址数组、size 数组，再调用 Ascend 的批量内存拷贝接口。

### 5.2 为什么需要批量 copy？

KV cache 的一个逻辑 block 可能对应多个 layer/state/tensor。如果每个小 tensor 单独提交 copy，会增加 Python、驱动和线程调度开销。批量 descriptor 允许一次提交多个地址范围，同时保持每个子 tensor 的 stride。

### 5.3 pinned memory

CPU offload 通常优先使用 pinned memory，使 NPU 与 host 之间可以更高效地发起 DMA。simple worker 在 pinned memory 不可用时会继续工作，但吞吐可能下降；这属于性能退化，不等同于语义失败。

---

## 6. 一次 save/load 的最小链路

### 6.1 save：NPU 到 CPU

```text
Worker 收到本轮 connector metadata
  -> 根据 block ids 找到 NPU source views
  -> 找到 CPU destination blocks
  -> launch_copy()
  -> NPU DMA thread 执行 batch copy
  -> 记录 job/transfer result
  -> get_finished() 报告发送完成
```

### 6.2 load：CPU 到 NPU

```text
Worker 收到需要恢复的 block ids
  -> 找到 CPU source blocks
  -> 找到 NPU destination views
  -> launch_copy()
  -> 等待或轮询完成事件
  -> get_finished() 报告接收完成
  -> forward 使用已恢复的 NPU KV block
```

native 路径的 handler 和 simple 路径的 copy backend 实现细节不同，但都遵守相同的上层 connector 生命周期：注册 cache、绑定 metadata、执行 load/save、等待完成、回传结果。

---

## 7. 哪些地方最容易误读？

### 7.1 offload 不等于 recompute

offload 保存完整 KV tensor，load 后可以直接使用；recompute CPU offload 保存的是重计算信息，恢复时可能重新执行计算。两者都可能使用 CPU，但恢复语义完全不同。

### 7.2 worker 不负责决定 token 命中

offload worker 只根据上层传来的 block/metadata 执行 copy。请求是否命中、需要多少 token、哪些 block 要分配，仍由 Scheduler、KVCacheManager 或 connector scheduler-side 逻辑决定。

### 7.3 buffer 可用不等于 copy 完成

CPU buffer 已分配、copy job 已提交，并不代表目标 NPU block 可以立即被 attention 使用。必须等待 transfer result 或对应同步点，再进入 forward。

---

## 8. 本文结论

```text
1. kv_offload 是本地 NPU/CPU tensor 搬运，不是外部 KV Store。
2. native 路径复用 vLLM 原生 offloading 抽象，Ascend 重点适配 layout 和 NPU worker。
3. simple 路径通过 SimpleCPUOffloadNPUWorker 和 NPUDmaCopyBackend 实现更直接的 block copy。
4. register_kv_caches() 建立 NPU tensor、CPU storage 和 block view 的关系，是两条路径的关键入口。
5. descriptor buffer、pinned memory 和批量 DMA 解决的是吞吐与地址组织问题，不改变 Scheduler 的 block 语义。
6. load/save 的最终完成状态仍需通过 connector 回到 vLLM 调度链。
```

下一篇建议阅读 [recompute_cpu_offload.md](recompute_cpu_offload.md)，分析“不保存完整 KV”时的重计算路径。
