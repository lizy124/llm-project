# [Ext] Backend 抽象基类拆分（Backend / GVABackend）

> 编号：kv-31 | 维度：Ext | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

`Backend` 基类定义了 `batch_alloc / batch_add_lease / batch_remove_lease / batch_get_key_info / batch_write_finish`，这些是 memcache GVA 专属概念，但放在基类里默认抛 `NotImplementedError`，污染了抽象。新增后端时基类方法会持续膨胀。

## 任务

拆分为：
- `Backend`（最小接口：put / get / exists / register_buffer）
- `GVABackend`（继承，增加 lease / alloc 接口）

worker 侧用 `isinstance` 或能力查询判断是否走 GVA 路径。

## 验收标准

### 1. 功能正确性
- 改动后 mooncake / memcache / yuanrong 三后端行为不变
- 现有单测全绿

### 2. 代码质量
- 基类不再因 GVA 概念膨胀
- 能力查询清晰

### 3. 交付件
- PR + 设计说明 + 单测

## 证据

- 五个 `batch_*` 接口默认抛 `NotImplementedError`：[backend.py:35-48](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py#L35-L48)
- `batch_is_exist` 透传不抛：[backend.py:32-33](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py#L32-L33)

## 重点关注

- 与 kv-26（backend put 失败传播）协同：失败语义在拆分后更易统一
- worker 侧 GVA 路径判断从"调用是否抛异常"改为"能力查询"更健壮

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
