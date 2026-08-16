# [Correctness] backend put 失败向上层传播，避免静默吞错

> 编号：kv-26 | 维度：Correctness | 严重程度：高 | 建议优先级：**P0**
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

`MooncakeBackend.put()` / `MemcacheBackend.put()` 失败时**只记录日志，不返回明确失败状态**；发送线程仍可能把请求标记为完成。后果是：写失败被静默吞掉，后续请求 / event 仍按"成功保存"处理，下游 get 命中空数据或读到 stale，影响正确性与可恢复性。

## 任务

让写失败的语义显式化：
1. backend `put()` 返回明确的成功/失败结果（状态码或异常），不再仅记日志
2. 发送线程（`KVCacheStoreSendingThread`）感知失败后，不把请求标记为正常完成，而是走异常清理路径
3. 失败可被上层（scheduler / worker）观测，触发降级（重算 / 标记 invalid block）

## 验收标准

### 1. 功能正确性
- 注入 put 失败（mock backend 返回失败 / 抛异常）后：
  - 请求不被标记为成功保存
  - 相关 block 进入 `_invalid_block_ids` 或等价失败标记
  - 下游 get 不会把失败 block 当作可用缓存
- 失败信息可在日志 / 状态中追溯

### 2. 回归保护
- 现有单测全绿
- 新增单测：put 失败 → 完成态不被错误设置 + 失败标记生效

### 3. 交付件
- PR + 设计说明（失败传播路径）
- 单测覆盖 mooncake / memcache 两条 backend 的失败分支

## 证据

- mooncake put：[mooncake_backend.py:189-221](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L189-L221)
- memcache put：[memcache_backend.py:210-238](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L210-L238)
- 发送线程仍进入完成态：[kv_transfer.py:697-715](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L697-L715)

## 重点关注

- 两个 backend 的失败语义需统一抽象（配合 kv-31 Backend 拆分）
- 与 kv-25（线程异常清理）联动：put 失败应走统一异常路径
- 注意部分失败（batch 中部分 key 失败）的细粒度处理

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
