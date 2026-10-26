# 读码问题、术语与地址模型

版本与证据统一见 [00](00_source_baseline.md)。本文件由旧随记收敛为复习索引；已核验的机制放入对应专篇，避免两处维护相互矛盾的长篇解释。

## 1. 术语对照

| 名词 | 本体系中的准确含义 |
|---|---|
| prefix hit | 某个请求状态下可跳过/恢复的前缀；需说明 local、external 或二者合并 |
| `kvpool_cached_tokens` | LoadSpec 的恢复目标水位；不能总当作“池自身总存量” |
| `kvpool_store_skip_tokens` | 已有外部对象对应保存跳过水位，可与计算水位不同 |
| matched / loaded | 前者是调度承诺，后者是数据实际恢复 |
| layerwise | 每层执行 load/save，不保证每层独立 key |
| block-key | 一个 block/group/stage 对象 + 多层 range |
| GVA | backend 对象寻址信息，不是本地 tensor pointer |
| session | backend 对象读写生命周期，须 commit/revoke/end |
| lease | 对象使用期间的位置/生命周期保护，不能由短 TTL hit cache 替代 |
| released fence | 本地账本已释放后，物理覆盖前等待旧保存 |
| event | 可能是 Python layer event、NPU stream event、distributed completed event，需说明种类 |
| PreemptOffload | 抢占时保存真实 KV/state；名称里的 recompute 语境不代表只存 metadata |

## 2. 复习问答入口

| 应能回答的问题 | 对应文档 |
|---|---|
| 插件怎样注册、为什么替换上游名字和 spec | [02 控制面](ascend_store/02_connector_and_control_plane.md) |
| 查到命中为何还要分配本地 block | [01 请求闭环](ascend_store/01_overview.md) |
| 为何 Scheduler/Worker 看似相同配置还会生成不同 key | [03 身份与布局](ascend_store/03_metadata_and_layout.md) |
| 为什么在 bind 时准备 layerwise，而不是等 start_load | [04 Worker](ascend_store/04_worker_pipeline.md) |
| Mooncake 逐层传输为什么不是每层一个对象 | [05 协议](ascend_store/05_transfer_backend_storage.md) |
| `wait_for_save` 为什么不一定已经完成保存 | [06 资源安全](ascend_store/06_concurrency_and_config.md) |
| Scheduler 能直连池，为何普通查询仍走 ZMQ | [07 查询](ascend_store/07_lookup_path_design.md) |
| 本地 hash 命中为何不能证明共享层还驻留 | [07 查询与复用](ascend_store/07_lookup_path_design.md) |
| target prefix 命中为何还不够 DSpark 恢复 | [08 组合场景](ascend_store/08_composition_dspark_kvpp.md) |
| get histogram 能否解释 layerwise TTFT | [09 指标与排障](ascend_store/09_observability_testing.md) |
| 抢占尾块没 hash，为什么还能恢复 | [PreemptOffload](other_paths/preempt_offload.md) |

## 3. base_addr 的正确心智模型

`tensor.data_ptr()` 是执行搬运使用的设备地址；`block_id` 是布局内索引。`ChunkedTokenDatabase.prepare_value()` 将二者结合：

```text
真实传输地址 = tensor base + block id × block stride
```

同一个逻辑层可以有多个 base entries；多层 views 也可共享一个底层 storage。因而：

```text
层数 != base_addr 数 != storage allocation 数 != backend 注册 region 数
```

- K/V 分离、scale、main/indexer/state 会增加 entries。
- alias/view 会使多个 entries 指向同一 allocation 的不同范围或相同范围。
- Worker 按 storage pointer 合并注册区间；注册数不一定等于层数。
- 物理 reuse 让多层轮转同一 slot，不等于这些层的数据同时保留。

只能从这里确认设备虚拟地址、tensor storage 与注册范围。底层物理页连续性、IOMMU/IOTLB、RDMA/HCCL 映射和吞吐效果，不能单靠这些 Python 地址推断。旧随记中“散页性能基本无感”等绝对判断不再作为源码结论保留。

## 4. 显存模型：只比较同一部分

KV buffer 可分为 owned persistent cache、共享 target scratch、layerwise reuse buffers 和其他状态；此外还有模型权重、运行 workspace、graph/通信等占用。

推导 buffer 复用收益时需固定：
1. 相同 page 数与 group/spec。
2. 相同 entry bytes 与 dtype。
3. 相同持久/临时 owner 范围。
4. 相同 partial、prefetch 和独立层配置。

“61 份 KV buffer 变 9 份”只能说明某个等大布局下的这部分 KV 容量，不代表整卡 HBM 使用变为 9/61，也不保证吞吐提升。

## 5. 分析结论的三个证据等级

- **代码事实**：当前分支、参数、返回值、等待点；写出文件和符号。
- **推导例子**：用简化页/层模型解释含义，明确前提。
- **运行结论**：真实环境中的精度、延迟、吞吐，需要两边 commit、SDK、拓扑与测量记录。

源码注释可帮助定位意图，但不能覆盖不同步的执行体；已有 UT 说明覆盖意图，也不等于真实后端验证通过。下一次更新按照 [00 的维护流程](00_source_baseline.md)复核即可。
