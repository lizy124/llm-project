# [Perf] GVA 元数据 RPC 跨 (request,group) 合并

> 编号：kv-08 | 维度：Perf | 严重程度：高 | 建议优先级：P1
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

`_alloc_gvas_for_save` 和 `_prepare_load_gvas` 收到的是**整批 requests**，但内部 `for request × for group` 双层循环里，每个 (request, group) 单独调 `batch_alloc`/`batch_is_exist`/`batch_get_key_info`/`batch_add_lease`。`batch_alloc` 非幂等、固定开销大，N×G 次可压成 1~2 次。

## 任务

收集所有 (request, group) 的 keys 拼成一次调用：
- `batch_is_exist`：所有 cached_keys 一次查
- `batch_alloc`：所有 new_keys 一次 alloc（sizes 是 list，按位置对应不同 group 的 alloc_size）
- `batch_get_key_info` / `batch_add_lease`：同理
- `_allocated_gvas` 是进程内 dict，合并查询后按偏移拆回各 request

## 验收标准

### 1. 功能正确性
- 合并后各 (request, group) 拿到的 gva / lease 与逐个调用一致
- partial block 单 key 场景不破坏
- 现有单测全绿

### 2. 性能验证
- `batch_alloc` 调用次数从 N×G 降到 1~2
- 一批 request 下的延迟对比

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- save 侧 per-(req,group) `batch_is_exist`：[pool_worker.py:1253](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1253)
- save 侧 per-(req,group) `batch_alloc`：[pool_worker.py:1278](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1278)
- load 侧 per-(req,group) `batch_get_key_info`：[pool_worker.py:1441](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1441)
- load 侧 per-(req,group) `batch_add_lease`：[pool_worker.py:1466](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1466)
- partial block 单 key：[pool_worker.py:1310](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1310)

## 重点关注

- key 命名空间全局唯一，跨 request 合并无冲突
- 合并后需按偏移正确拆回各 request 的 `_allocated_gvas`
- 与 kv-07 模式相同，可同步推进
- 前置依赖 kv-33（补测试）

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
