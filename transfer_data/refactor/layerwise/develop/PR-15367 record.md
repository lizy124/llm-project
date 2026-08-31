# PR #15367(refactor_layerwise_part1)前后梳理

> 代码:`refactor_layerwise_part1` @ `735065fe1`(5 commits,基于 `9c3cf949d`)
> 基线:`9c3cf949d` = 合入 #15291 后的 upstream/main
> 规模:13 files,+433/−62,零行为变化
> 系列:layerwise GVA 重构第一批(前身 PR-A #15277 被本 PR 取代,见 §6)

## 1. 前因

时间线上的四个节点决定了这个 PR 的形状:

**#15307(前身的合并 PR)。** layerwise GVA 传输线程与协议代码原散落在
`kv_transfer.py`(通用传输层)与 `pool_worker.py`/`pool_scheduler.py`
中,GVA(memcache 专属协议)知识与通用数据面知识混在同一批文件里。
#15307 一次性完成全部收敛(314 个单测通过),但改动量大、验证周期长,
评审与服务器验证成本高。结论:不直接用,改为从 upstream/main 开新分支
分批实施,每批独立可验证、可合入。

**#14465(线上回归)。** GVA gate 的派生式
`use_layerwise and backend == "memcache"` 在仓库里有 4 份副本
(worker / scheduler / layout / connector)。#14465 清理死代码时删掉了
connector 侧那份,但 `AscendStoreConnector.set_external_slot_release_waiter`
仍然读它——MultiConnector(PD 分离部署)初始化直接 AttributeError。
根因:派生逻辑复制多份后,没有任何机制保证副本之间一致,"哪份是权威"
没有答案,删任何一份都可能炸。

**#15291(热修)。** 在 connector `__init__` 恢复那 2 行派生副本,
成为第 5 份。热修正确但不解决结构性问题:下一轮死代码清理仍可能重演
#14465。

**use_gva_layerwise 暴露问题。** 热修之后,connector(本应是薄转发层)
重新持有 GVA 协议知识。用户要求:该逻辑收敛进 backend 包,
`ascend_store_connector.py` 中不再出现。

## 2. 意图与边界

第一批只做"核心逻辑收敛",刻意不做大面积搬迁:

| 做 | 不做(留给后续批次) |
|---|---|
| 派生单点化(CAP) | GVA 传输线程类搬入 backend |
| key 构造统一(KEY) | worker 内 8 处 `self.use_gva_layerwise` 分支下沉 |
| gate 从 connector 下沉 | metadata/kv_transfer 的 GVA 段落整体迁移 |
| 基线替换:删 #15291 热修副本 | 任何行为语义变化 |

判定标准:每个提交单独 checkout 出来,运行行为与 main 一致。

## 3. 设计决策

### 3.1 CAP:能力表 + 单点函数(`backend/__init__.py`)

形态:`_BACKEND_CAPABILITIES` 静态表(memcache → `{gva_layerwise}`,
其余为空)+ `use_gva_layerwise(use_layerwise, backend_name)` 函数。

**为什么是"字符串进、布尔出"的静态函数**:调用点
`layerwise_cache_layout.get_gva_layerwise_config()` 发生在 KV cache 布局
构建期,此时 connector / worker / backend 实例均未创建,手上只有
`kv_connector_extra_config` 里的字符串——布局期没有任何实例可查询,这是
根本约束。静态表让这一刻的回答无需 import 任何 backend 模块(被否的
类属性方案恰要先拉起 `memcache_backend` 才能读到属性;torch 布局期本就
已就绪,"不可拉起"不成立,不作为论据)。

**为什么放 `backend/__init__.py`**:`backend_map`(名字→模块路径)已
在此,backend 能力属于同类注册表知识。

**代价与对策**:静态表是第二事实源,会与类实现漂移。
`test_backend.py` 的一致性测试锁定两侧:表声明 `gva_layerwise`
⇔ 该类通过 MRO override 了全部 5 个 GVA 方法。

**被否方案**:能力改为类属性(`Backend.GVA_LAYERWISE_CAPABLE`)只
覆盖 worker/scheduler 两处(那里可拿到实例),布局处仍需静态表,
变成两套机制,弃。

### 3.2 IFACE:基类不动 + 线程入口断言(经历过两次改向)

这条决策演化了三步,最终形态是三步中最小的:

1. **ABC 方案(初版,已回滚)**:从 `Backend` 基类删除 5 个
   `NotImplementedError` 桩,新建 `GVALayerwiseCapable` ABC,
   `MemcacheBackend` 双继承。动机是让 mypy 对门控调用点提供静态保护。
   回滚原因:评审观点——基类原有的东西不该删,基类桩是 main 的既有
   接口形状,删除属于接口收紧而非纯收敛。
2. **方案 A 讨论(未完全采纳)**:讨论过 A(isinstance 收窄到
   `MemcacheBackend`,保留静态保护)/ B(mypy cast,零保护)两路,
   A 优于 B 的判据是 #14465 型回归的特征——静态不可见、运行时才炸。
   采纳了 A 中的运行时断言部分。
3. **最终形态**:base.py 与 main 逐字节一致(5 个桩留在 `Backend`
   上,`MemcacheBackend` 单继承);layerwise 两个线程的
   `_handle_request` 入口加
   `assert isinstance(self.m_store, MemcacheBackend)`。
   语义:线程只在 gate 开启时启动,不变量成立时断言恒真(零开销);
   不变量被破坏时,线程入口立即 AssertionError,而非首次 store 调用
   延迟抛 NotImplementedError——错误定位从"某个方法缺实现"提前为
   "gate 不变量被破坏"。

### 3.3 KEY:`GVAKeyFactory`(`backend/gva_protocol.py`)

worker 侧与 scheduler 侧各有一份 key 字符串构造(拼法相同但代码独立),
格式漂移会把命中变成未命中,且是跨集群的线上数据格式。收敛为一个
工厂类的三个静态方法:

- `full_key`:单组 `model@hash@rank`(PR #11585 兼容格式),
  多组 `model@group@hash@rank`
- `partial_key`:`model@partial@req@group@idx@end_token@rank`
- `hit_check_keys`:每个 head_or_tp_rank 一把 full key

`test_gva_protocol.py` 用逐字节快照锁定格式(期望值从重构前的
pool_worker / pool_scheduler 实现转录)。同函数
`get_partial_block_index` 从 pool_worker 迁至 `metadata.py`。

### 3.4 gate 下沉(connector 纯转发,取代 #15291)

`AscendStoreConnector.set_external_slot_release_waiter` 变纯转发,
gate 移至 `KVPoolWorker.set_external_slot_release_waiter`:worker 已经
持有单点派生的 `self.use_gva_layerwise`,gate 在数据面消费者处判定,
返回 bool 表示 waiter 是否被接受。

connector `__init__` 中 #15291 恢复的 2 行派生随之删除——本 PR 与
#15291 是取代关系:热修恢复的副本恰是本 PR 要消灭的第 5 份。
`use_gva_layerwise` 标识符在 `ascend_store_connector.py` 中仅存于
docstring(说明设计约束),代码引用为零。

## 4. 提交结构

每个提交独立行为保持,按逻辑依赖排序:

| # | commit | 内容 | 文件 |
|---|---|---|---|
| 1 | `6334f638e` | backend 能力表 + `use_gva_layerwise` 单点(仅新增,无调用方切换;能力表测试在提交 2 一并落地) | backend/__init__.py |
| 2 | `34de1fc01` | layerwise 线程入口 `assert isinstance(m_store, MemcacheBackend)`(净新增,不动 base.py) | kv_transfer.py, test_backend.py |
| 3 | `42fa31654` | GVAKeyFactory 平移两份 key 构造 + `get_partial_block_index` 迁 metadata;worker/scheduler 两处派生切单点函数;worker `set_external_slot_release_waiter` 加 gate、返回 bool(此刻唯一调用方 connector 仍持 #15291 gate 且忽略返回值,行为不变) | gva_protocol.py(新), metadata.py, pool_worker.py, pool_scheduler.py, 两个测试 |
| 4 | `9f804019b` | layout 派生切单点函数(worker/scheduler 已在提交 3 切换);UT stub 包执行真实 backend/__init__,补提交 2/3 的 stub 缺口 | layerwise_cache_layout.py, _mock_deps.py |
| 5 | `735065fe1` | gate 下沉:connector 纯转发 + 删 #15291 副本(gate 与 bool 返回已在提交 3 落地) | ascend_store_connector.py, test_ascend_store_connector.py |

提交 5 的 message 完整记录 #14465 → #15291 → 取代的因果链;PR 描述补
一行 "Supersedes the connector-side flag restored by #15291 (gate moved
to the worker)"——diff 中删除的 2 行即刚合入的热修副本,一句话说明来源
即可,其余背景不在 PR 描述展开。

补正记录(2026-08-31):提交 3 的 message 原漏记 worker gate hunk,已
amend 补记;C3-C5 hash 随之更新(`765042e79`→`42fa31654`、
`ec3878b57`→`9f804019b`、`345a4ecea`→`735065fe1`),最终树与补正前
逐字节一致(`git diff 345a4ecea 735065fe1` 为空)。

走读复核补记(2026-08-31,详见 `PR-15367 walkthrough.md` §3.7/§4.3):

- 提交 3 实际还包含 worker/scheduler 两处派生切换 hunk(原表格记在
  提交 4,已按事实修正);提交 4 实际仅改 layout + _mock_deps 两个
  文件,其 message 的 "at all call sites" 以 hunk 论失实
- 提交 2/3 单独 checkout 时 ascend_store UT 不可运行(stub 手工
  mirror 缺 `_BACKEND_CAPABILITIES`/`use_gva_layerwise`,
  ImportError),提交 4 的 exec 修改回填——"每提交独立可验证"仅在
  生产行为层面成立,UT 层面断档;PR CI 只测 head 故无感知
- 上述 message 级偏差是否再 amend(需再 force-push)由作者定夺,
  树级内容不改

## 5. 测试覆盖

| 文件 | 覆盖 |
|---|---|
| `test_backend.py` | 能力真值表(memcache/mooncake/yuanrong/大小写/False 组合);表↔MRO override 一致性;未知 backend / 未知 capability 返回 False |
| `test_gva_protocol.py` | 三类 key 的字节级快照;hit_check 与 full_key 的 rank 格式一致性 |
| `test_ascend_store_connector.py` | `set_external_slot_release_waiter` 转发契约:worker gate 拒绝(非 GVA backend)/接受两路;connector 不持有 flag 的回归守卫 |
| `test_pool_worker.py` | worker gate 的接受/拒绝路径 |

## 6. 与相关 PR 的关系

- **#15277**:前身 PR-A(2026-08-29 提交,Open,含 `GVALayerwiseCapable`
  ABC 方案,与本 PR "base.py 不动"的最终形态方向相反)。评审后弃用该
  实施、改为从 main 分批,本 PR 取代其 part1 范围;本 PR 合入时需关闭
  #15277 或在其上标注 superseded,避免两个方向相反的 PR 并存
- **#15307**:前身合并 PR,已搁置不用(其验证清单被本 PR 复用)
- **#14465**:回归源头,本 PR 使该类回归在结构上不可能发生(派生只有一份)
- **#15291**:热修,本 PR 删除其 connector 侧副本,属取代关系
- **#12711**(Open,412b157 亦修 #14465 回归):与本 PR 提交 5 处置同一处
  connector 代码;若其先合入,本 PR rebase 时按"删 connector 侧派生、
  gate 在 worker"的本 PR 意图解决冲突
- **#14697**(本作者,死代码清理后续,Open):与本 PR 无硬依赖,交叠
  冲突预计为机械性,后合入者 rebase

## 7. 遗留与后续(part2 方向)

- worker 内 8 处 `self.use_gva_layerwise` 读取(pool_worker.py:381/423/
  472/487/525/830/1167/1333)仍在调用方分支判定,能力知识尚未完全沉入
  backend 接口
- GVA 传输线程类(`kv_transfer.py` 内 LayerSendingThread /
  LayerRecvingThread 等)仍在通用传输层文件中,与 GVA 协议的物理收敛
  未完成
- `gva_protocol.py` 目前只含 key 工厂,协议其余部分(租约管理、
  write-finish 语义)仍散在 worker / 线程代码里
- 提交 2 为线程入口断言在 `kv_transfer.py` 顶部引入了 `MemcacheBackend`
  直接 import——通用传输层重新持有具体 backend 引用,与收敛方向相反;
  part2 搬迁线程类时应随之消除(断言随线程类移入 backend 侧)

服务器验证项见同目录 `server-validation.md`(与 #15307 清单的 1/2/3
相同;验证项 1 关注点不同:gate 在 worker 侧)。
