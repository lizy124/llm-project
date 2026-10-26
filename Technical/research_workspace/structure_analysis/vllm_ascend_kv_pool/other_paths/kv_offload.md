# Native / Simple KV Offload：本地内存层级搬运

基线见 [00](../00_source_baseline.md)。本篇分析 `kv_offload/native/` 与 `kv_offload/simple/`；抢占专用路径见 [PreemptOffload](preempt_offload.md)。

## 1. 与 AscendStore 的对象边界

| 维度 | AscendStore | Native / Simple offload |
|---|---|---|
| 定位 | 外部 KV pool | 本地 accelerator/CPU 层级 |
| 数据面身份 | key、GVA、object range | 本地 page/block 映射与 tensor view |
| 控制面 | Store scheduler/coordinator | 对应上游 offload scheduler/manager |
| 设备适配 | 注册 buffer、backend I/O | canonical cache、NPU copy handler |
| 一致性 | commit/write finish、lease/session | copy event、job result、buffer 生命周期 |

上游 offload 的 scheduler 本身仍可按 hash 维护缓存；“不走 AscendStore 外部 key 协议”不等于“没有 prefix/hash 语义”。native tiering 还可复用上游 secondary tiers，因此不能把整个 native 系统一概说成“只有一块 CPU buffer”。

## 2. 注册同时替换 connector 与 spec

`distributed/kv_transfer/__init__.py::register_connector()`：

```text
OffloadingConnector -> AscendOffloadingConnector
CPUOffloadingSpec -> NPUOffloadingSpec
TieringOffloadingSpec -> NPUTieringOffloadingSpec

SimpleCPUOffloadConnector -> AscendSimpleCPUOffloadConnector
  （上游 simple 模块可导入时）
```

native wrapper 保留上游 scheduler、manager、metrics 与 transfer lifecycle；Ascend 重点替换 worker canonicalization 和 accelerator 搬运。

## 3. Native：先把设备 cache 转成 canonical pages

源码：
- `native/offloading_connector.py`。
- `native/npu.py`。
- `native/cpu_npu.py`。
- 上游 `kv_connector/v1/offloading/`、`vllm/v1/kv_offload/`。

上游 Worker 需要统一的页表示；Ascend 可能暴露 split K/V、scale、Mamba state、多 kernel pages 或带 stride 的 view。

`AscendOffloadingConnectorWorker.register_kv_caches()`：
1. 判断是否需要 Ascend layout adaptation；标准布局继续走上游。
2. 使用上游 canonical mappings 保持 group/logical page 对齐。
3. 用 `_canonicalize_split_cache()` 识别真实 component 覆盖。
4. 用 `_make_int8_block_view()` 构造零拷贝 `[num_blocks, page_bytes]` 字节页。
5. 形成 `CanonicalKVCaches` 的 tensors/refs/page mappings，交给 spec/worker。

### 3.1 不能只按 shape 选择“最大 view”

一个 view 看似覆盖完整页，却未必实际包含 V 或 scale 的首块地址。代码检查 underlying storage 和地址范围；无法由一个 full view 覆盖时，保留多个 components。

合并多个 physical pages 成一个 manager page 时，必须确认 inner payload 连续、相邻页不重叠、需要拼接的页相邻。错误布局会明确拒绝，不能用 reshape 掩盖真实 stride。

### 3.2 CPU-only 与 tiering spec

| spec | 内存与 Worker |
|---|---|
| `NPUOffloadingSpec` | PyTorch pinned host tensors，创建 `NPUOffloadingWorker`；不把任意 mmap 假设为可按 CUDA 方式 host-register |
| `NPUTieringOffloadingSpec` | 通过 `SharedOffloadRegion` 连接 Worker 与 scheduler-side primary tier，复用上游 secondary tier 机制 |

tiering 的 replicated layout 只在上游认可的字节复制拓扑下使用；其 rank/共享 mmap 区域选择也不同于 CPU-only 路径。

## 4. Native copy handler

`SingleDirectionNPUOffloadingHandler` 按 job 收集 src/dst/size descriptor，通过 Ascend 批量 copy 与 NPU stream/event 提交异步操作。

```text
上游 store/load specs
  -> NPUOffloadingWorker.submit_store / submit_load
  -> 单方向 handler.transfer_async
  -> event / TransferResult
  -> get_finished / wait
```

descriptor buffer 不能在设备仍使用时复用；job 已提交不表示 copy 已结束。`wait(job_ids)` 与 `shutdown()` 各自要处理在途任务。

CPU pinned allocator 与 shared region 是不同内存资源，不能仅因都位于 host 就混用注册与生命周期假设。

## 5. Simple：保留上游 scheduler，替换 NPU Worker

`AscendSimpleCPUOffloadConnector` 先运行上游初始化，再替换 `worker_handler`。上游在此阶段只建立 Python 状态，设备资源延后到 cache registration，所以这种替换不会先创建一套 CUDA transfer 资源。

**prefix caching 被关闭时，上游可让 worker 保持未设置，使 connector 成为 no-op；Ascend 保留这个条件。**

`SimpleCPUOffloadNPUWorker.register_kv_caches()` 建立：
- 逻辑 block 对应的设备 views。
- CPU storage 和 views。
- 每 component 的 page bytes/stride。
- `NPUDmaCopyBackend` 与复制参数。

## 6. Simple DMA 后端

`simple/copy_backend.py::NPUDmaCopyBackend` 在后台线程消费 copy jobs，使用专用 NPU stream，调用 `npu_mem_ops.copy_blocks()` 发起批量搬运，记录可轮询的 event。

```text
launch_copy(src_blocks, dst_blocks)
  -> queue
  -> copy thread / dedicated NPU stream
  -> copy_blocks + BatchMemcpyParams
  -> completion event
  -> 上游 worker/connector 汇总
```

`BatchMemcpyParams` 描述所有 component 的字节数、stride 与地址展开关系；一个 logical block 不保证等于一个连续 tensor。CPU buffer 分配成功、job 入队、event 完成是三个阶段。

## 7. 三条本地路径如何选择阅读

| 路径 | 核心问题 |
|---|---|
| native | 如何在上游通用 offloading/tiering 体系接入 Ascend layout 与 NPU DMA |
| simple | 如何用较直接的 CPU block buffer 和 copy backend 复用上游 simple 生命周期 |
| preempt | 如何在请求被抢占时保存真实 KV，含未 hash 的 partial tail，并在恢复时精确对齐状态 |

测试入口：
- `tests/ut/kv_offload/test_native_cpu_offload.py`。
- `tests/ut/kv_offload/test_npu_offload_spec.py`。
- `tests/ut/simple_kv_offload/test_simple_kv_offload.py`。
- `tests/e2e/pull_request/one_card/test_cpu_offloading.py` / `test_simple_cpu_offload.py`。

完整测试索引见 [09](../ascend_store/09_observability_testing.md)。
