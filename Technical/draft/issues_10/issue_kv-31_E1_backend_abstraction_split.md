# [Ext] 显式建模 AscendStore backend capability

> 编号：kv-31 | 维度：Ext | 严重程度：低到中 | 建议优先级：P3
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

`Backend` 除通用 `set_device/register_buffer/exists/put/get` 外，还定义了 `batch_get_key_info`、`batch_alloc`、`batch_add_lease`、`batch_remove_lease` 和 `batch_write_finish` 等 GVA/lease 能力，默认抛 `NotImplementedError`；目前只有 MemcacheBackend 实现这些能力。

当前 scheduler/worker 通过 `use_layerwise and backend_name == "memcache"` 选择 GVA 路径，backend 注册表也只包含 Mooncake、MemCache 和 Yuanrong。因此当前没有独立的用户配置可以选择“GVA mode + 非 GVA backend”；Mooncake/Yuanrong 的 layerwise 配置走 key-based 路径。这里能确认的是基类接口混杂和未来扩展风险，不能描述成已经可达的生产故障。

`batch_is_exist` 是三个 backend 都使用的通用批量 exists，不应归入 GVA 专属接口。

## 任务

1. 保留最小通用 `Backend`，显式定义普通模式需要的基础能力。
2. 引入内部 `GVABackend` ABC 或 runtime-checkable protocol，包含 allocation、key info、lease 和 write-finish 契约；MemcacheBackend 实现该能力。
3. scheduler client、worker backend、lazy initialization 和 transfer thread 在进入现有 MemCache GVA 路径时完成类型窄化/能力校验；不新增用户配置，也不改变 Mooncake/Yuanrong layerwise 路由。
4. 用缺失 capability 的 fake/future backend 构造内部错误组合，验证一旦代码选择 GVA 路径会在初始化边界 fail-fast，并给出包含 mode/backend/missing capability 的明确错误。
5. 保存经过窄化的 backend 引用，避免在热路径散落 `isinstance` 或通过捕获 `NotImplementedError` 判断能力。

## 验收标准

### 1. 功能正确性
- Mooncake、MemCache、Yuanrong 的普通模式行为不变
- MemCache GVA scheduler lookup、allocation/lease、write-finish 和 release 全路径通过 capability 校验
- 缺失 capability 的 fake/future backend 一旦被内部路由到 GVA 路径，会在初始化期失败；不把它表述为当前已有用户配置
- scheduler-only backend、worker backend 和 lazy init 均被覆盖
- `batch_is_exist` 继续作为通用能力使用

### 2. 代码质量与测试
- GVA 专属方法不再污染最小 Backend 契约
- capability check 集中在初始化边界，热路径不重复判断
- 参数化测试覆盖三个 backend 的类型关系，以及 fake/future backend 的缺失 capability 构造
- 现有单测全绿

### 3. 交付件
- PR + capability/interface 说明 + 单测

## 证据

- backend contracts：`backend/base.py:9-56`
- MemCache implementation：`backend/memcache_backend.py:41-238`
- scheduler：`pool_scheduler.py:169-180, 344-390`
- worker：`pool_worker.py:145-156, 306-323, 1184-1535`
- transfer threads：`kv_transfer.py:1308-1652`

## 重点关注

- 本任务是内部类型边界清理，不是已发生的 P0 正确性问题
- 不为制造“错误配置”测试而增加当前不存在的 GVA 用户开关
- 不通过 backend 名称硬编码替代 capability 校验
- 与 kv-26 的 `put` 结果契约可独立推进

## 环境约定
- vllm-ascend：审核基线 `d5e9816065ede613327d93908f87fee9f5c47128` 或提交时最新 main
- 硬件：Ascend NPU（注明型号、卡数和 backend）
- 关联任务池：#9079
- 验收人：@赵鹏博
