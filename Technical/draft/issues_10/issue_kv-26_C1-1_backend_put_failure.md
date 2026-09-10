# [Reliability] 统一 Backend.put per-key 结果与 save 可观测性

> 编号：kv-26 | 维度：Reliability | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

Mooncake、MemCache 和 Yuanrong 的普通 `put` 路径会吞掉 batch 返回失败或 binding 异常，只记录日志并返回 `None`。Backend 抽象没有统一返回契约，sender 无法知道哪些 key 真正写入；启用 KV events 时还可能为失败 key 发布 `BlockStored`。

`finished_sending` 的主要语义是本地 GPU block 已读取完、scheduler 可以释放 block，并不等同于所有远端 key 已持久化。保存失败通常应表现为后续 cache miss，而不是把 producer request 或 block 标为 load invalid。因此本任务聚焦可靠性、事件准确性和可观测性，不复用 consumer 的 `_invalid_block_ids` 通道。

不同 backend 的原始返回码未必都是“0 成功、非 0 失败”：重复对象/已存在对象可能已经可用于后续 lookup，Yuanrong binding 也可能只提供整批异常而没有逐 key 返回。具体语义不能从当前 wrapper 静态推断。

## 后端契约前置验证

1. 固定实际 SDK/服务版本，分别验证全成功、重复/已存在、部分失败、整批异常和返回长度异常时的原始返回值及后续 `exists/get` 可用性。
2. 根据验证结果定义规范化状态，至少区分 `AVAILABLE`（新写入或已可用）、`FAILED` 和 `UNKNOWN`，并保留 raw backend code。不能简单把所有非零码判失败，也不能把没有逐 key 证据的整批调用伪造成逐 key 成功。
3. 若某 backend 不提供逐 key 结果，评估 post-write `exists` 是否语义可靠且成本可接受；否则允许返回 `UNKNOWN`，此时不发布 `BlockStored`，但仍完成本地 block 生命周期。

## 任务

1. 为普通 `Backend.put` 定义与输入 keys 等长的标准 per-key 结果，至少区分 `AVAILABLE`、`FAILED` 和 `UNKNOWN`，并承载 backend/raw code 或异常类型。
2. Mooncake、MemCache、Yuanrong 统一实现该契约，并校验返回长度。
3. sender 只为确认 `AVAILABLE` 的 key 生成 KV stored events；`FAILED/UNKNOWN` key 记录结构化状态、错误码、计数和 backend 维度指标。
4. 无论远端持久化是否成功，本地 block 读取完成后都继续完成 block 生命周期，避免错误地阻止 scheduler 释放 GPU blocks。
5. 为连续/系统性失败预留明确的 backend health 或 circuit-breaker 接口；是否使 connector unhealthy 需单独定义阈值，不能由单 key 失败隐式触发。

## 验收标准

### 1. 功能正确性
- 三种 backend 的 `put` 返回值长度和 key 顺序一致
- 全成功、部分失败和 binding 异常均可逐 key 定位
- KV events 只包含经后端契约确认已可用的 key；`UNKNOWN` 不得被当作 stored
- save 完成通知仍能释放本地 blocks，不因远端失败长期挂起 producer request
- 后续 lookup 对失败 key 表现为 miss，不制造虚假的 stored 状态

### 2. 可观测性与回归
- 日志/指标包含 backend、失败类型和失败 key 数量，并避免逐 key 日志风暴
- 新增三 backend 的全成功、部分失败、异常和返回长度错误测试
- 现有单测全绿

### 3. 交付件
- PR + Backend.put 契约说明 + 指标说明 + 单测

## 证据

- Backend contract：`backend/base.py:50-56`
- Mooncake：`backend/mooncake_backend.py:189-221`
- MemCache：`backend/memcache_backend.py:210-238`
- Yuanrong：`backend/yuanrong_backend.py:147-160`
- sender/events：`kv_transfer.py:679-715, 807-892`

## 重点关注

- GVA layerwise save 已检查 `batch_copy`/`batch_write_finish`，不属于本任务的普通 put 契约
- 重复/已存在对象是否属于 `AVAILABLE` 必须由对应 backend 契约或实验确认
- 单 key 保存失败不是模型正确性 P0，也不应写入 load-side invalid block 集合
- 与 kv-25 对齐系统性 backend fatal 的传播边界，但两项可以独立实现

## 环境约定
- vllm-ascend：审核基线 `d5e9816065ede613327d93908f87fee9f5c47128` 或提交时最新 main
- 硬件：Ascend NPU（注明型号、卡数和 backend）
- 关联任务池：#9079
- 验收人：@赵鹏博
