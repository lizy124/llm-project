# [Perf] MLA non-layerwise 读取的 TP 内后端访问去重 PoC

> 编号：kv-01 | 维度：Perf | 严重程度：中高 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

MLA 下 `num_kv_head=1`。代码审计可以确认：在 non-layerwise 同步读取路径中，每个 TP rank 当前都会独立构造目标地址并调用 backend `get`；同一 TP 组在其余 key 维度一致时会查询相同 key 集。这里能确认的是“调用形态重复”，不能仅凭静态代码确认真实 backend 是否做了等价去重，也不能确认重复读取是当前 TTFT 瓶颈。

这与写侧行为不能直接类比：non-layerwise 写侧按 `put_step` 对 block 分片，并非只有 rank 0 写；layerwise/GVA 又有独立的 buffer、event 和传输状态机。因此本任务只验证 non-layerwise 路径，不预设 `leader get + TP broadcast` 一定优于各 rank 独立读取。

## 开发前置验证（Phase 0）

实现 collective PoC 前，先在目标 Ascend 环境完成以下验证：

1. 对 TP=2/4/8 记录各 rank 最终 key 集、backend `get` 次数、每次 item 数和读取耗时，确认候选场景中 key 集确实一致；如存在差异，先给出适用条件，不能直接广播。
2. 分解独立读取的 backend 耗时和一个等量 TP broadcast 的耗时，至少覆盖一个高后端延迟和一个低后端延迟环境。
3. 预先约定 Go/No-Go 阈值。若重复读取占比很低、broadcast 明显更贵或目标 collective 无法安全承载错误传播，可仅提交验证报告并判定暂不实现/不启用；这属于有效结论。

## 任务

1. Phase 0 达到 Go 条件后，实现可开关的 PoC：由 TP leader 执行 backend `get`，再通过现有 NPU collective 向 TP 组广播读取结果。
2. 明确 leader 选择、目标 buffer 布局、stream/event 同步和 backend 部分失败的传播语义。
3. 保持非 MLA 路径不变；layerwise/GVA 不纳入本任务。
4. 只有端到端数据达到预先定义的收益阈值时才考虑默认启用；未达到阈值时保持现状并记录结论，不为完成任务强行引入 broadcast。

## 验收标准

### 1. 功能正确性
- TP=2/4/8 下，各 rank 最终 KV 数据与逐 rank `get` 逐项一致
- backend miss、部分失败和异常能够传播到全部参与 rank，不产生 collective hang
- greedy / non-greedy 输出一致，非 MLA 与 layerwise/GVA 行为不变
- 现有单测全绿

### 2. 性能验证
- 报告 TP=2/4/8、prompt 16K/64K 下 backend 调用次数、读取耗时、broadcast 耗时、TTFT 和吞吐
- 至少覆盖高后端延迟与低后端延迟两类环境，说明收益边界
- 对比独立读取与 leader+broadcast，不预设后者必然更快

### 3. 交付件
- Phase 0 验证报告；达到 Go 条件时再交付 PR + PoC 设计说明 + 性能数据 + 单测

## 证据

- MLA/key-head 派生：`pool_worker.py:195-205`、`metadata.py:36-58`
- non-layerwise 同步读取：`pool_worker.py:888-980`
- non-layerwise 写侧分片：`pool_worker.py:1026-1031`、`kv_transfer.py:788-797`

## 重点关注

- collective 必须由所有 TP rank 以一致顺序进入，错误路径不能只让部分 rank 提前返回
- broadcast 成本可能高于重复 backend 读取，结论必须由硬件实测决定
- 不再包含与 MLA 不兼容的 TP mismatch 验收项

## 环境约定
- vllm-ascend：审核基线 `d5e9816065ede613327d93908f87fee9f5c47128` 或提交时最新 main
- 硬件：Ascend NPU（注明型号、卡数和 TP）
- 关联任务池：#9079
- 验收人：@赵鹏博
