# PR #15367(refactor_layerwise_part1)前后梳理

> 代码:`refactor_layerwise_part1` @ `ee6220d7c`(5 commits,基于 `40f9834ee`)
> 基线:`40f9834ee` = upstream/main(2026-08-31;原基线 `9c3cf949d`,因
> #15386 revert 触发的 CI 基线漂移而 rebase,见下方 rebase 记录)
> 规模:12 files,+431/−66,零行为变化(生产行为;测试适配见提交 2)
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
| 派生单点化(GATE 入 memcache 协议模块) | GVA 传输线程类搬入 backend |
| key 构造统一(KEY) | worker 内 8 处 `self.use_gva_layerwise` 分支下沉 |
| gate 从 connector 下沉 | metadata/kv_transfer 的 GVA 段落整体迁移 |
| 基线替换:删 #15291 热修副本 | 任何行为语义变化 |

判定标准:每个提交单独 checkout 出来,运行行为与 main 一致。

## 3. 设计决策

### 3.1 GATE:单点函数入 memcache 协议模块(`backend/gva_protocol.py`)

最终形态:裸函数 `use_gva_layerwise(use_layerwise, backend_name)`,
内部 `strip().lower()` 归一化后与 `"memcache"` 比较,与
`GVAKeyFactory` 同住协议模块;`backend/__init__.py` 回归纯注册表
(仅 `backend_map`,本 PR 零 diff)。

**为什么是"字符串进、布尔出"的静态函数**:调用点
`layerwise_cache_layout.get_gva_layerwise_config()` 发生在 KV cache 布局
构建期,此时 connector / worker / backend 实例均未创建,手上只有
`kv_connector_extra_config` 里的字符串——布局期没有任何实例可查询,这是
根本约束。函数所在模块零 import,布局期 import 它不拉起任何 backend
实现。

**为什么进协议模块而非 backend 类 / `__init__.py`**(三堵墙):
布局期无实例(类方案要求静态映射兜底,两套机制);GVA 的编排
(session/租约/hit-check)跨 worker/scheduler,塞进 backend 类违反
"协议不持有 worker 状态";`memcache_backend.py` 顶部 import torch,
协议知识并入会拖重依赖、破坏 UT stub 的纯度。`backend/__init__.py`
曾为候选(与 `backend_map` 同址),被否:领域函数污染包入口,且其
零 import 是无防护约定(明天谁加一个 re-export 就崩),独立模块的
零 import 是结构性保证。

**排他性的测试化**:`test_gva_protocol.py::TestGvaMemcacheExclusivity`
直接断言"5 个 GVA store 方法仅被 MemcacheBackend override"——
"GVA 是 memcache 专属协议"从 docstring 升级为可执行断言。

**形态演化(同日三步,如实记录)**:能力注册表
(`_BACKEND_CAPABILITIES` + `backend_supports`,住 `__init__.py`)→
采纳 Gemini 归一化意见(head `16bdb52fd`)→ 识别到"专属协议"事实后
压扁为裸函数、表消亡、迁入协议模块(head `ee6220d7c`)。中途形态
未经历 CI,仅存在于当日 force-push 间隙。泛化机械(表+查询函数)为
"未来第二能力"预留,事实不需要,压扁后护栏不丢(排他性直连断言)。

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
| 1 | `7b9e1e530` | `gva_protocol.py` 新建:`use_gva_layerwise` 单点(内部归一化);gate 真值表 + memcache 排他性测试 | gva_protocol.py(新), test_gva_protocol.py(新) |
| 2 | `234419dce` | layerwise 线程入口 `assert isinstance(m_store, MemcacheBackend)`(净新增,不动 base.py);`test_kv_transfer.py` 两个线程 fixture 改 `spec=MemcacheBackend` mock 以过断言 | kv_transfer.py, test_kv_transfer.py |
| 3 | `1dc6daae3` | GVAKeyFactory 平移两份 key 构造 + `get_partial_block_index` 迁 metadata;worker/scheduler 两处派生切单点函数;worker `set_external_slot_release_waiter` 加 gate、返回 bool(此刻唯一调用方 connector 仍持 #15291 gate 且忽略返回值,行为不变) | gva_protocol.py, metadata.py, pool_worker.py, pool_scheduler.py, 两个测试 |
| 4 | `83cbf9402` | layout 派生切单点函数(worker/scheduler 已在提交 3 切换);UT stub 包执行真实 backend/__init__ 保持 backend_map 同步 | layerwise_cache_layout.py, _mock_deps.py |
| 5 | `ee6220d7c` | gate 下沉:connector 纯转发 + 删 #15291 副本(gate 与 bool 返回已在提交 3 落地) | ascend_store_connector.py, test_ascend_store_connector.py |

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
  生产行为层面成立,UT 层面断档;PR CI 只测 head 故无感知。
  **此断档已被当日第三次重塑消除**:gate 函数迁入 `gva_protocol.py`
  后,子模块经 stub 包 `__path__` 解析(无需 exec),C3' 上用旧
  stub 实测 151 passed——"每提交独立可验证"重新在 UT 层面成立
- 上述 message 级偏差是否再 amend(需再 force-push)由作者定夺,
  树级内容不改(注:第三次重塑时 message 已随新形态重写,该条
  针对 `16bdb52fd` 及更早形态的历史记录)

CI 失败与修复记录(2026-08-31,head `735065fe1` 首轮 CI):

- 首轮 CI 7 failed / 2826 passed,全部在 `test_kv_transfer.py`
  的 7 个 layerwise 线程测试:提交 2 的入口
  `assert isinstance(self.m_store, MemcacheBackend)` 打挂了既有
  测试的裸 `MagicMock()` store fixture
- 根因:push 前本地验证未覆盖 `test_kv_transfer.py`(教训:入口
  加 isinstance 断言时,必须排查所有直接调用 `_handle_request`
  的既有测试);本地复现与 CI 完全一致(非环境差异)
- 修复:两个线程 fixture 改 `MagicMock(spec=MemcacheBackend)`
  (PR-A b89884b 既定模式);`.store` 为实例属性、spec 限制属性
  访问,需显式 `store.store = MagicMock(batch_copy=...)` 装配。
  修复 amend 进提交 2(断言属提交 2,适配同属),提交 2-5 hash
  更新(`34de1fc01`→`f1f96928d`、`42fa31654`→`a1f4427a5`、
  `9f804019b`→`5c840cf4c`、`735065fe1`→`bfeaacb14`),相对
  `735065fe1` 的树差异仅 `test_kv_transfer.py`(+13/−4)
- 修复后本地全量 ascend_store UT:283 passed + 2 failed
  (test_coordinator 两例为本地 stub 既有失败,同 PR-A 时代,
  CI 有真实 vllm 可过);ruff check/format 通过

Rebase 记录(2026-08-31 第二轮,head `bfeaacb14` CI):

- 第二轮 CI 在测试选择阶段失败:`select_tests.py: error:
  unrecognized arguments: --pr-labels`。非 PR 代码问题(PR 未碰
  `.github/`)
- 根因是 upstream 基线漂移:#15386(revert #14198)把 `--pr-labels`
  从 main 的 select_tests.py 与 pr_test.yaml **两边同时删除**,
  main 自身自洽;但 `pull_request` 事件的 workflow YAML 取自 PR head
  (旧基线 `9c3cf949d`,仍在传 `--pr-labels`),而 checkout 的
  merge commit 中脚本取 main 新版(已删该参数)——旧 workflow ×
  新脚本必然炸。凡基线早于 #15386 且未 rebase 的 PR 全部中招
- 处置:rebase 到 `40f9834ee`(main 最新)。基线间仅 8 个提交且
  零个触碰 PR 文件面,零冲突;PR 自身 diff 与 rebase 前完全一致
  (14 files,+446/−66),UT 结果不变(283 passed + 2 既有失败)。
  5 个提交 hash 全部更新(`6334f638e`→`af30d1d20`、
  `f1f96928d`→`d424aa6f1`、`a1f4427a5`→`ab31c1b8a`、
  `5c840cf4c`→`4e2c72384`、`bfeaacb14`→`9f5c199ea`),已 force-push

Gemini Code Assist review 处置(2026-08-31,review 于 03:04 UTC 生成,
针对旧版代码——评论引用的 `GVALayerwiseCapable` 是已废弃的 PR-A ABC
方案;实质意见 2 条):

- **采纳:backend_supports 内部归一化**(strip+lower)。消除"调用方
  必须先 lower"的隐式契约(layout 调用点不查 backend_map,新调用方
  漏 lower 会静默 False——#14465 型静默失效温床);时机上该函数是
  本 PR 新 API,合入后再改即二次破坏。三个现有调用点本就全部先
  lower,生产行为零变化。配套:真值表 `MEMCACHE` 翻转为 True 并补
  ` Memcache ` 用例,新增 `test_backend_supports_normalizes_name`
- **不采纳:assert 换 raise TypeError**。理由:ascend_store 模块既有
  74 处 assert(含本 PR 未动的代码),assert 是本库的不变量惯例;
  `python -O` 剥断言后仍有 base 桩 NotImplementedError 兜底,失败
  只延迟不消失;mypy 收窄两种写法等价。已在 PR 回复说明
- 实施方式:软重置到 `40f9834ee` 后按 C1-C5 分组重建提交链(期间
  发现 stash/amend 路径 parent 错位,推倒重来,最终逐 commit stat
  复核)。相对 `9f5c199ea` 树差异恰好为 Gemini 修复(+14/−2,仅
  `backend/__init__.py` 与 `test_backend.py`);UT 284 passed(新增
  1 例)+ 2 个 coordinator 既有失败;ruff 通过。hash 更新:
  `af30d1d20`→`441b48cf3`、`d424aa6f1`→`6ecb899e9`、
  `ab31c1b8a`→`12d312803`、`4e2c72384`→`e4009383c`、
  `9f5c199ea`→`16bdb52fd`,已 force-push 并在 PR 评论触发
  `/gemini review` 对新 head 重审

第三次重塑:GVA 归位 memcache 协议模块(2026-08-31,head `16bdb52fd`
→ `ee6220d7c`):

- 触发:作者走读后裁定"GVA 是 memcache 专属协议"是铁的事实,
  能力注册表是为不存在的"第二能力"预留的泛化机械;且 GVA 领域
  函数不应住在 `backend/__init__.py`(污染包入口,零 import 从
  约定变结构的诉求)
- 改动:`_BACKEND_CAPABILITIES` 表与 `backend_supports` 消亡;
  `use_gva_layerwise` 压扁为裸函数(保留归一化)迁入
  `gva_protocol.py` 与 `GVAKeyFactory` 同住;一致性测试改为直连
  断言(5 个 GVA 方法仅被 MemcacheBackend override);gate 相关
  测试全部并入 `test_gva_protocol.py`;`backend/__init__.py` 与
  `test_backend.py` 回归基线(退出 PR diff 面,14→12 files)
- 附带收益:旧形态的"C2/C3 中间态 UT 断档"消失(见走读复核补记)
- 实施:最终树先在 head 上完整构建并全量验证(UT 282 passed +
  ruff),固化为 TEMP 提交取树 hash `888ce6beb`;软重置到基线后
  按 C1'-C5' 分组重放(gva_protocol/test_gva_protocol 两文件按
  gate-only → final 两阶段构造);重建后树 hash 与 TEMP 逐字节
  一致,且 C1'/C2'/C3' 均当场跑过 UT(吸取首轮 CI 教训)
- GitHub 侧:PR 描述重写(突出 memcache-exclusive 语义与
  `backend/__init__.py` 回归纯净);补 follow-up 评论说明
  `backend_supports` 已随重构消亡(前一条回复中的名称不再存在)

## 5. 测试覆盖

| 文件 | 覆盖 |
|---|---|
| `test_gva_protocol.py` | gate 真值表(memcache/mooncake/yuanrong/大小写/空白/False 组合);未知 backend 返回 False;memcache 排他性(5 个 GVA 方法仅被 MemcacheBackend override);三类 key 的字节级快照;hit_check 与 full_key 的 rank 格式一致性 |
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
- `gva_protocol.py` 目前含 gate 函数与 key 工厂,协议其余部分(租约
  管理、write-finish 语义)仍散在 worker / 线程代码里(part2 的
  GVASession/GVAHitChecker 归宿已定:同模块)
- 提交 2 为线程入口断言在 `kv_transfer.py` 顶部引入了 `MemcacheBackend`
  直接 import——通用传输层重新持有具体 backend 引用,与收敛方向相反;
  part2 搬迁线程类时应随之消除(断言随线程类移入 backend 侧)

服务器验证项见同目录 `server-validation.md`(与 #15307 清单的 1/2/3
相同;验证项 1 关注点不同:gate 在 worker 侧)。
