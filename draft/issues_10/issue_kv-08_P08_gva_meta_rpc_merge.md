# [Perf] 单次 process_layer_data 批次内聚合 GVA 元数据 RPC

> 编号：kv-08 | 维度：Perf | 严重程度：中高 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

`_alloc_gvas_for_save` 和 `_prepare_load_gvas` 接收整批 requests，但当前在 request/group 循环内分别调用 `batch_is_exist`、`batch_alloc`、`batch_get_key_info` 和 `batch_add_lease`。减少 MemCache metadata RPC 次数可能降低 layerwise 准备阶段开销，但实际调用次数取决于 new keys、partial blocks、cache hit 和 lease 状态，不是固定的 N×G。

full-block key 不包含 request ID，相同前缀请求可能共享 key；`batch_alloc` 非幂等，lease 还可能具有引用计数或状态机语义。聚合前必须先明确去重和回填规则，不能直接把所有 key 拼成一次调用。

## 任务

采用两阶段实现：

1. 为当前 `process_layer_data` 批次构造 descriptor，记录 request/group、full/partial key、alloc size、block index 和回填位置。
2. 对 `batch_is_exist`/`batch_get_key_info` 等只读查询按后端允许的规则去重，并把结果映射回全部使用位置。
3. `batch_alloc` 只对确认不存在的唯一 full keys 调用；partial key 保持 request scope。验证返回长度、GVA 有效性和部分失败。
4. 在确认 MemCache lease 语义后决定 `batch_add_lease` 是否可去重；保留 partial lease 的 `MMC_UNMATCHED_STATE` 重试、已取得 lease 的 rollback 和最终 release 对称性。
5. 合并 kv-10 的有效部分：descriptor 构造期间只转换实际使用的 block hash，并在同一 request/group 范围复用字符串结果；不为该局部转换单独承诺性能收益。

## 验收标准

### 1. 功能正确性
- 重复 full key 正确共享查询/allocation 结果，并回填到所有 request/group
- partial block、eviction、allocation failure、lease failure/retry 和 multi-group rollback 行为不变
- RPC 返回长度异常或部分失败时 fail-fast，不能留下半初始化 GVA 或未释放 lease
- save/load descriptor 的 block hash 字符串和 key 与改动前逐项一致
- 现有单测全绿

### 2. 性能验证
- 报告各 metadata API 在 batch size 1/4/8/16 下的调用次数和 item 数
- 报告 descriptor 构建耗时、GVA 准备耗时和 TTFT；不预设一定降到一次调用
- 分别覆盖高重复 key 和低重复 key workload

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- save/GVA allocation：`pool_worker.py:1184-1365`
- load/GVA metadata 与 lease：`pool_worker.py:1367-1569`
- 当前批次入口：`pool_worker.py:1651-1670`
- lease release：`kv_transfer.py:1633-1640`

## 重点关注

- full-block key 可跨 request 重复，必须显式去重和多位置回填
- `batch_alloc` 非幂等；lease 是否允许去重必须由 MemCache 契约或实验确认
- `_allocated_gvas` 是进程内状态，更新必须与 batch 结果和 eviction 检查一致
- 与 kv-07 模式相同，可同步推进
- 本任务已吸收原 kv-10 的局部重复 hash-to-string 清理

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
