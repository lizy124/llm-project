# [Perf] 单次 connector step 内批处理 non-layerwise backend 调用

> 编号：kv-07 | 维度：Perf | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

non-layerwise 路径存在减少 backend 调用次数的机会：同步 load 按 request 调用 `get`；async recv thread 每次处理一个 request；普通 save thread 则按 request/group 执行 `exists` 和 `put`。三个 backend 的接口都接受批量 key/address/size 列表。

但跨 request 合并不能假设 key 唯一。full-block key 不包含 request ID，相同前缀请求可能生成重复 key；不同 request 还拥有独立的 block IDs、store/load mask、完成状态、失败结果和 NPU event。合并实现必须保存逐项归属，不能只拼接列表后统一标记成功。

## 任务

1. 建立单次 connector step 的 transfer descriptor，记录每项所属 request/group、key、block ID、address、size、event 和回填位置。
2. 写侧先批量查询 exists，再只为 missing 项构造最终 `keys/addrs/sizes`；可去重 exists 查询，但必须定义重复 key 的 put 和逐 request 完成语义。
3. 同步读侧只合并当前 step 已知的 requests，并把 backend 的 per-item 结果准确映射回 request/block。不要在合并后的全局列表上改变原有 per-request circular-shift 语义。
4. async recv 如需微批，只能采用有数量和等待时间上限的 bounded batch；不能无限 drain queue，避免单请求尾延迟和取消语义失控。
5. 合并 kv-12 的有效部分：lookup 返回后直接从 descriptor/missing indices 生成最终参数，减少中间列表重建；不要为 Python 字符串和嵌套地址引入 NumPy object array。

## 验收标准

### 1. 功能正确性
- 合并前后每个 request 的 key、目标地址、失败 block、完成状态和 KV event 一致
- 重复 full-block key、partial block、multi-group、store/load mask、skip-null 和 TP mismatch 行为正确
- backend 部分失败时只影响对应 request/block，不能把整个 batch 错标成功
- request 取消、preemption 和 thread fatal 不遗留无法完成的 batch 项
- 现有单测全绿

### 2. 性能验证
- 提供 batch size 1/4/8/16 的 backend 调用次数、batch 构建耗时、TTFT/吞吐和尾延迟
- 分别测同步 load、async load 和 save，不预设所有路径都能降到一次调用
- 只有 profile 显示端到端收益时才默认启用；否则保留简单路径或设置合理阈值

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- 同步 load：`pool_worker.py:888-1017`
- 普通 save thread：`kv_transfer.py:679-715, 717-890`
- async recv thread：`kv_transfer.py:923-1039`
- thread queue/fatal 语义：`kv_transfer.py:496-525`

## 重点关注

- full-block key 可跨 request 重复，partial key 才包含 request ID
- backend 的最大 batch、线程安全和失败返回契约需要逐 backend 验证
- 与 kv-25/kv-26/kv-28 的失败传播语义对齐，但不依赖独立测试任务
- 本任务已吸收原 kv-12 中可随批处理实现清理的中间列表重建问题

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
