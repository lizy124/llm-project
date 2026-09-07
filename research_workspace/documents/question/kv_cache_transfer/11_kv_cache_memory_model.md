# 11. KV Cache 的显存模型：VA、物理页与 base_addr

源码位置：

- `vllm/vllm/v1/worker/gpu_worker.py`
- `vllm/vllm/v1/worker/utils.py`
- `vllm/vllm/utils/mem_utils.py`
- `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
- `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py`

本问题关注：vLLM 拉起服务时 GPU/NPU 显存被谁占用；KV cache 的大小如何推导、何时分配；base_addr 从哪里来、是逻辑编号还是真实地址；"连续 VA 段"与"物理页散落"如何同时成立；散落的物理页对 RDMA/HCCL 注册和传输性能有何影响。

---

## 1. 启动时显存被谁占用

```text
模型权重          静态，常驻，TP/PP 切分后每 rank 持有分片
KV cache          动态，占比最大，启动时一次性预分配
激活/临时缓冲      forward 中间张量、logits/sampling buffer
CUDA graph        capture 的静态输入张量 + graph 结构
通信缓冲          NCCL/HCCL buffer、KV transfer staging buffer
框架固定开销       CUDA context / driver 保留区、allocator 碎片保留
```

## 2. KV cache 大小如何推导

源码链路（v1 路径）：

- 预算 = `total_memory × gpu_memory_utilization`，**不是 free × util**：
  `vllm/v1/worker/utils.py` 的 `request_memory()`；
  启动时物理 free 不足该值则直接 `raise ValueError`，服务拉不起。
- profile 阶段：`memory_profiling()`（`vllm/utils/mem_utils.py`）包住 `profile_run()`，
  进出各执行一次 `gc.collect() + torch.accelerator.empty_cache()`。
- 最终大小（`vllm/v1/worker/gpu_worker.py`）：

```text
available_kv_cache_memory
  = requested_memory - non_kv_cache_memory - cudagraph_estimate

non_kv_cache_memory
  = 权重 + profile 峰值激活 + 非torch（NCCL 等）
```

时序要点：

```text
CUDA context 建立
→ 加载权重
→ memory_profiling（前后 empty_cache）
→ 算 num_gpu_blocks → 分配 KV cache（之后运行期不再 malloc）
→ capture CUDA graphs / sampler warmup
```

- KV cache 预分配是静态快照：运行期不扩容，耗尽走 preemption。
- 启动后其他进程释放显存，vLLM 不会感知。
- 唯一例外是 sleep/wake 机制（显式释放、唤醒重申请）。

## 3. "散落的空闲显存"会不会分给 KV cache

会。只要拉起那一刻物理上 free：

- `init_snapshot.free_memory` 来自 `get_memory_info`，是驱动视角的全局 free，
  不关心历史碎片；
- profile 前后的 `empty_cache()` 已把进程内临时碎片归还驱动；
- KV cache 大 tensor 走全新大分配，所有 free 物理页可用。

拿不到的只有三种：别的进程占的、context/driver 保留的、
`empty_cache` 时仍被活跃张量持有的。

## 4. base_addr 是什么

**不是逻辑编号，是真实设备地址**，来自 `cache.data_ptr()`：

- 采集：`pool_worker.py` 的 `register_kv_caches()` / `_infer_cache_group_metadata()`
  遍历每层 cache tensor，逐个 `data_ptr()` 存入 `group_kv_caches_base_addr`。
- 注册：`set_group_buffers()`（`metadata.py`）原样接收。
- 使用：`prepare_value()` 纯算术寻址：

```python
addr = base_addr + block_id * block_stride
```

严格说是设备虚拟地址（VA），由 NPU SMMU 翻译到物理地址；
block_id 才是逻辑的（block table 里的索引）。

## 5. 60 层 = 60 个 base_addr？

账面上是：base_addr 条目数 = 层数 × 每层 cache entry 数。

| 模型/后端                | 每层 entry 数 | 60 层的 base_addr 数 |
|--------------------------|---------------|----------------------|
| MLA/DSV3（单 latent）     | 1             | 60                   |
| DSV4（main + indexer）    | 2             | 120                  |
| MHA K/V 分开后端          | 2             | 120                  |

但 entry 数 ≠ 物理内存块数。这 60/120 个 tensor 经常是同一块大 flat
buffer 的 views：

- vLLM 侧 `_allocate_kv_cache_tensors()` 按 packed 条目分配 backing，多层 alias。
- ascend_store 侧 `_get_storage_key()` 取 `untyped_storage().data_ptr()` 做 key，
  注册 MR 时同 storage 的地址 min/max 合并成一个 region。

60 个 base_addr 是寻址粒度（layerwise 传输定位"第 N 层的 block M"），
物理上是同一块预分配显存的不同偏移。

## 6. 连续 VA 段 ≠ 连续物理内存

VA 本质上只是编号：相邻地址编号代表"逻辑上相邻的字节"，不承诺物理相邻。
`torch.zeros(40G)` 时发生的事：

```text
第 1 步  向驱动要 40G 地址空间
        → VA 无限（64bit），划一段连续编号 [0x100000000, +40G)
第 2 步  驱动从 HBM 空闲池找物理页（典型 2MB 一页）
        → 哪有空放哪，不要求相邻
第 3 步  页表记录映射
```

页表形态（示意）：

```text
虚拟地址(编号)                物理位置(HBM)
──────────────────────────────────────────
0x100000000 + 0~2MB     →   物理页 @ 0x5F8000000
0x100000000 + 2~4MB     →   物理页 @ 0x212000000    ← 跳到很远
0x100000000 + 4~6MB     →   物理页 @ 0x5F8200000    ← 跳回来贴着第一页
    ...                            ...
```

左边连续、右边不连续，两句描述同时成立，因为说的是同一块东西的两面。

每个使用者都不需要物理连续：

| 谁在访问            | 怎么找到物理内存              | 关心物理连续吗 |
|---------------------|-------------------------------|----------------|
| NPU 算子            | load/store 走 SMMU 翻页表     | 不关心         |
| base_addr+stride    | 纯 VA 算术                    | 不涉及         |
| RDMA/HCCL DMA       | 注册时驱动替它翻页表生成 SGL   | 不要求         |

60 个 `data_ptr()` 是同一段连续 VA 编号里的 60 个不同起点，
`base_addr + block_id × stride` 自始至终只在 VA 上做加法。

一个佐证：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
（CUDA VMM）能 "remap a virtual address range to different physical
pages"（`vllm/config/vllm.py` `_verify_kv_transfer_compat()` 的注释）——
能 remap 恰恰说明 VA 背后的物理页本来就是页粒度映射、可散布的。

心智模型：页表是图书馆索引卡——书架号连续排（VA 连续），
每张卡指向的书在仓库哪个格子由仓库当时哪有空决定（物理散落）。
所有操作只跟卡片打交道，只有搬运工真正去仓库时才按卡片找格子。

## 7. 散页对 RDMA/HCCL 性能的影响

注册时（一次性）：

- `reg_mr` 走页表、pin 页、生成 SGL。物理页越碎 SGL 越长。
- 实际形态是 **2MB 粒度散落**（NPU/GPU 大块分配天然大页背书），
  不是 4K 粒度：几十 GB MR 的 SGL 段数是几千，注册成本百 ms 量级。
- MR 启动时注册一次、全生命周期复用，不进稳态路径。

传输时（稳态）基本无感：

- PCIe 事务本来就是 TLP 粒度（payload 典型 256B~4KB），
  物理连续不可能"一次搬几十 MB"，散页与连续页在这层无区别。
- RDMA 引擎对 SGL 流水线处理，段开销摊到 MB 级数据块可忽略。
- 2MB 段背书下 IOTLB 条目少、命中率高。

代码里已经在做的对的事：

- `pool_worker.py` `_align_kv_ptrs`：注册区起点向下对齐 2MB。
- `pool_worker.py`：同 storage 多层地址 min/max 合并成一个 region，
  MR 数量从"层数"降到"storage 数"。
- `sparse_kv_offload_manager.py`：`_CPU_CACHE_ALIGNMENT = 2MB`，
  CPU 侧 pin memory 同样按 2MB 对齐。

真正会差的情况：4K 粒度真散页（大块分配不会出现）、
频繁 reg/dereg、IOTLB 容量小 + 工作集巨大导致 TLB 抖动。

## 8. expandable_segments 为什么被禁止

`vllm/config/vllm.py` 的 `_verify_kv_transfer_compat()`：
配置了任何 KV connector 时显式 `raise ValueError` 拒绝
`expandable_segments:True`（除非启用 cumem allocator）。

原因：VMM 会把 VA remap 到不同物理页，已注册的 MR
（ibv_reg_mr、Mooncake、ascend_store 的 registered_regions）
会指向 stale 物理页，第一次 RDMA 就报
`IBV_WC_REM_ACCESS_ERR` / `NIXL_ERR_REMOTE_DISCONNECT`。

结论：走 RDMA 注册的路径必须放弃动态重映射能力——
对 pinned 内存，地址稳定比碎片治理重要。

---

## 9. 一句话心智模型

```text
KV cache 显存 = 启动时一次大分配的连续 VA 段，
物理页由驱动页表映射、可散落；
block_id 是逻辑索引，base_addr 是 VA，
base + id × stride 在 VA 上算术寻址；
RDMA 注册 pin 的是 VA 段背后的物理页，2MB 粒度散落无碍性能。
```
