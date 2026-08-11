# [Perf] MLA 读侧去重：rank0 取 + TP broadcast

> 编号：kv-01 | 维度：Perf | 严重程度：高 | 建议优先级：P1（已有方案）
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

MLA 写侧已去重（只 rank 0 写），但**读侧未去重**：每个 TP rank 各自从池子 get 一份到各自 buffer。TP 越大浪费越多，且 MLA 小数据固定开销占比高。已有专项方案文档。

## 任务

实现读侧去重：rank 0 取 + TP 组 broadcast，对齐写侧的 `put_step` 去重模式。

## 验收标准

### 1. 功能正确性
- 去重后各 TP rank 拿到的数据与去重前逐字节一致
- TP mismatch 路径不破坏去重逻辑
- MLA / 非 MLA 模型均不回归

### 2. 性能验证
- 给出 TP=2/4/8 下，改动前后 get 调用次数与端到端延迟对比
- 长序列（16K/64K）下的收益数据

### 3. 交付件
- PR + 设计说明
- 单测 + 性能数据

## 证据

- 写侧去重：[pool_worker.py:1020](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1020) `if self.tp_rank % self.put_step != 0: return`
- 读侧无对应跳过：[pool_worker.py:867-980](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L867-L980)
- 专项方案：[MLA_KV读取去重优化讨论.md](file:///D:/lzy/project/kv_pool/llm-project/draft/MLA_KV读取去重优化讨论.md)

## 重点关注

- broadcast 路径与现有 NPU collective 的复用
- 注意 `put_step` 与读侧 step 的语义对齐
- 与 kv-07（非-layerwise I/O 合并）模式不同，避免混淆

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数 + TP）
- 关联任务池：#9079
- 验收人：@赵鹏博
