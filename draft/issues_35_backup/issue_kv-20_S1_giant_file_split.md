# [Refactor] 巨文件拆分（pool_worker/kv_transfer/config_data/scheduler）

> 编号：kv-20 | 维度：Refactor | 严重程度：高 | 建议优先级：P1
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

4 个文件超过 1000 行，职责过重，影响可维护性和后续所有改动：
- `pool_worker.py`（2306 行）：`KVPoolWorker` 承担初始化、load、save、GVA 分配、TP mismatch、线程管理、lookup server 等多职责
- `kv_transfer.py`（1535 行）：三类传输线程（非layerwise / key layerwise / GVA layerwise）混杂
- `config_data.py`（1112 行）：既放数据类（KeyMetadata/PoolKey/LayerPoolKey），又放业务逻辑（ChunkedTokenDatabase），还放 metadata（ReqMeta/RequestTracker/LoadSpec）
- `pool_scheduler.py`（1106 行）：scheduler 编排 + lookup client 混杂

## 任务

按职责拆分：
- `config_data.py` → `keys.py`（PoolKey 及相关）+ `token_database.py`（ChunkedTokenDatabase）+ `request_meta.py`（ReqMeta/RequestTracker/LoadSpec）
- `pool_worker.py` → `load.py` / `save.py` / `gva.py` / `tp_mismatch.py`，`KVPoolWorker` 只做编排
- `kv_transfer.py` → 按线程类型拆分（非layerwise / key layerwise / GVA layerwise 各一文件）

## 验收标准

### 1. 功能正确性
- 拆分后行为完全不变（纯结构重构，零语义改动）
- 现有单测全绿（必要时调整 import 路径）
- 公共 API（对外暴露的类/函数签名）不变

### 2. 代码质量
- 每个新文件职责单一，行数控制在合理范围
- import 关系清晰，无循环依赖

### 3. 交付件
- PR + 拆分映射表（旧文件行段 → 新文件）
- 必要时补 import 兼容层（过渡期）

## 证据

- `pool_worker.py:2306` / `kv_transfer.py:1535` / `config_data.py:1112` / `pool_scheduler.py:1106`
- 见概览表：[kv_pool_优化点系统性梳理.md](file:///D:/lzy/project/kv_pool/llm-project/draft/kv_pool_优化点系统性梳理.md)

## 重点关注

- **前置依赖 kv-33（补测试）**：无测试网重构风险极高
- **前置依赖 kv-24（配置 schema）**：拆分时配置读取应已集中
- 与 kv-02（key 向量化）协同：先拆 keys.py 再做向量化
- 拆分建议分批 PR，避免单 PR 过大难 review

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
