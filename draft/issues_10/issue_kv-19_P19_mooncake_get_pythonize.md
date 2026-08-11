# [Perf] Mooncake get 返回值降 Python 化

> 编号：kv-19 | 维度：Perf | 严重程度：中高 | 建议优先级：P1
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

`MooncakeBackend.get()` 会把 C++ 返回值转成 Python list，再扫描失败码、再扫描一次把正值归零。对于大批量 get，这是稳定的 O(N) Python 成本（对象化 + 两遍扫描）。对比 memcache 也存在结果扫描，但未做同样的 list 化与二次处理。

## 任务

让后端直接返回更紧凑的状态结构，或只在失败分支才展开详细码，减少 Python 层对象化和重复遍历。

## 验收标准

### 1. 功能正确性
- 返回状态语义与改动前一致（成功 / 失败码）
- 现有单测全绿

### 2. 性能验证
- 大批量 get（blocks 大）下 Python 层耗时下降（profile 对比）
- 端到端延迟收益

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- [mooncake_backend.py:240-256](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L240-L256)
- 对比 memcache：[memcache_backend.py:185-198](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L185-L198)

## 重点关注

- 与 kv-09（消除 tolist）同属"numpy/ndarray 在 C++ 边界前的 Python 化"主题
- 与 kv-31（Backend 抽象拆分）协同：状态返回结构可统一抽象

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
