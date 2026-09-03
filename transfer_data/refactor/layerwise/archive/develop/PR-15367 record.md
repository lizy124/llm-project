# PR #15367(refactor_layerwise_part1)前后梳理

> 代码:`refactor_layerwise_part1` @ `6fc433ea2`(5 commits,基于 `e8f47fc11`;
> 2026-09-01 工作区另有一批**未提交**改动——协议解析收敛的 part1 提前量,
> 4 files +45/−19,已备份分支 `backup-pre-protocol-bind` @ `6fc433ea2`,
> 实施状态与遗留见 §7.3 / §7.4)
> 基线:`e8f47fc11` = upstream/main(2026-09-01;历经两次 rebase:首次因
> #15386 revert 的 CI 基线漂移从 `9c3cf949d`→`40f9834ee`,第二次为
> 重触发 CI 刷新 flake 判断→`e8f47fc11`,见下方记录)
> 规模:15 files,+497/−76,零行为变化(生产行为;测试适配见提交 2)
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

### 3.1 GATE:注册表数据绑定(`backend_map` 的 `layerwise_protocol` 字段)

最终形态:memcache 条目携带协议模块路径,通用层经两个中性解析函数
问答,从不按名 import 协议模块:

```python
# backend/__init__.py(唯一出现 memcache→GVA 绑定的地方)
"memcache": {..., "layerwise_protocol": "...backend.gva_protocol"},

def backend_supports_layerwise(backend_name: str) -> bool: ...   # 归一化后查字段
def get_layerwise_protocol(backend_name: str): ...               # 懒加载协议模块
```

```python
# 三个消费者(全部零 GVA 字样)
# worker / scheduler:
self.use_gva_layerwise = use_layerwise and backend_supports_layerwise(self.backend_name)
# layout(get_gva_layerwise_config 更名 get_layerwise_reuse_config):
if extra_config.get("use_layerwise", False) and backend_supports_layerwise(backend_name):
```

**设计的核心:绑定写在数据里,不写在代码里**。依赖反转的实现载体是
`backend_map` 条目的一个字段——通用层问"这个 backend 带不带 layerwise
协议",而不是"这是不是 memcache/GVA"。`use_gva_layerwise` 函数消亡,
其职责由字段存在性 + 归一化查询承载。

**与被否能力表的三个本质差异**:能力表为"未来第二能力"投机预留;
字段本身就是反转机制(不用它,通用代码就得硬编码 memcache 字符串,
退回泄漏)。能力表是扩展点;字段是依赖反转的造价,每个反转都长这样。

**互锁护栏**:`test_backend.py::TestLayerwiseProtocolRegistry` 断言
"条目带字段 ⇔ 类 override 全部 5 个 GVA 方法"+"字段指向的模块可解析
且含 GVAKeyFactory"+ 真值表(含 MEMCACHE/带空白归一化)。

**剩余泄漏(如实盘点,part2 范围)**:worker/scheduler 仍 import
`GVAKeyFactory`(key 构造,运行期 GVA 编排——协议对象的领域);kv_transfer
的 `MemcacheBackend` assert(脚手架);worker 内 8 处 `use_gva_layerwise`
分支与 base.py 5 桩。part1 消掉的是 gate 的全部 import 泄漏
(3 处 → 0)。

**演化全记录(同日四步,如实)**:能力表(`__init__`)→ 裸函数入
gva_protocol(Gemini 归一化采纳)→ 双参签名(dict-in 被数据流证伪)→
注册表字段绑定(依赖箭头反转,本次)。每次都由一个具体质疑推动:
泛化投机→入口污染→抽象边界→依赖方向。

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
| 1 | `bfd6f3354` | `gva_protocol.py` 新建:`use_gva_layerwise(use_layerwise, extra_config)` 单点(backend 键/默认值/归一化内置);gate 真值表 + memcache 排他性测试 | gva_protocol.py(新), test_gva_protocol.py(新) |
| 2 | `6bb71ce78` | layerwise 线程入口 `assert isinstance(m_store, MemcacheBackend)`(净新增,不动 base.py);`test_kv_transfer.py` 两个线程 fixture 改 `spec=MemcacheBackend` mock 以过断言 | kv_transfer.py, test_kv_transfer.py |
| 3 | `e75fc891e` | GVAKeyFactory 平移两份 key 构造 + `get_partial_block_index` 迁 metadata;worker/scheduler 两处派生切单点函数(双参);worker `set_external_slot_release_waiter` 加 gate、返回 bool(此刻唯一调用方 connector 仍持 #15291 gate 且忽略返回值,行为不变) | gva_protocol.py, metadata.py, pool_worker.py, pool_scheduler.py, 两个测试 |
| 4 | `7f3f7e31a` | layout 派生切单点函数(worker/scheduler 已在提交 3 切换);UT stub 包执行真实 backend/__init__ 保持 backend_map 同步 | layerwise_cache_layout.py, _mock_deps.py |
| 5 | `1ff8dc141` | gate 下沉:connector 纯转发 + 删 #15291 副本(gate 与 bool 返回已在提交 3 落地) | ascend_store_connector.py, test_ascend_store_connector.py |

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

第二轮 CI(16 卡 e2e flake)与第二次 rebase(2026-08-31 → 09-01):

- head `ee6220d7c` 的 CI:29/30 checks 绿(ruff/mypy/全量 UT/其它
  e2e),唯一失败 `a3-16 card-(part 1-1)` 的
  `test_kimi_k3.py::test_k3_mla_pd_tp8`——600s 服务器就绪超时,
  判定环境 flake(证据:该 job 在本 PR 旧 head `735065fe1` 曾全绿;
  测试用 `MooncakeConnectorV1`,与 ascend_store 零 import 关联;
  失败签名是基建超时而非断言,且 conftest `_wait_for_multiple_servers`
  按 host 键控 ready 导致 `not_ready: []` 掩盖真实未就绪方——
  captured log 显示 prefill 侧 50971 未就绪)。已在 PR 贴两条评论
  (分析 + 更正未就绪方为 prefill)
- 处置:`gh run rerun` 无 admin 权限,改为 rebase 重触发。main 前进
  18 个提交(kv_transfer 侧变更均在 `kv_p2p/`/`sparse_kv_offload/`,
  与 PR 的 ascend_store 面零重叠),零冲突;PR diff 面与 UT 结果
  完全不变(12 files +431/−66;282 passed + 2 coordinator 既有失败)。
  hash 更新:`1685f508a`/`abd351f34`/`ee5e9ab4b`/`935bb019c`/
  `2ff5cc890`,已 force-push

签名重定:gate 改双参(2026-09-01,head `2ff5cc890` → `1ff8dc141`):

- 触发:作者走读 layout 调用点,指出旧签名下"backend 键名/默认值/
  str+lower 抽取"仍留在调用方,抽象边界欠一档;先试纯 dict-in
- dict-in 被 3 个 UT 失败证伪:`use_layerwise` 在 worker/scheduler 是
  **构造参数**(connector 读 config 后传入),不是 config 键——
  纯 dict-in 把 gate 信任链从参数换成 config,与 `self.use_layerwise`
  (驱动 10+ 处非 GVA layerwise 行为)形成隐性双源
- 定稿双参 `(use_layerwise, extra_config)`:flag 是调用方权威值,
  backend 键/默认值/归一化归函数——知识的完整归属以数据流事实为界
- 实施:TEMP 树固化(hash `3b4d22022`)→ 分组重放;首轮重放 C3 的
  cherry-pick 带入旧标量调用(树 hash 不符 + 125 failed 当场暴露),
  以 TEMP 树文件版本修正后树 hash 逐字节一致;全量 UT 282 passed +
  137 subtests,ruff 通过,C3 中间态复验 151 passed 无断档。
  hash:`bfd6f3354`/`6bb71ce78`/`e75fc891e`/`7f3f7e31a`/`1ff8dc141`,
  已 force-push

步骤 1-2 提交完成(2026-09-01,head `6fc433ea2` → `2a239d18a`,7 提交):

- **批A(CI 阻断修复)**:`test_layerwise_cache_layout.py` 仍 import
  改名前的 `get_gva_layerwise_config`,整文件收集失败(CI 上一轮
  失败签名);改为 `get_layerwise_reuse_config` 并补 memcache 未
  opt-in 用例。原计划 amend 进 C4;执行时远端已被推送不含该修复的
  5 提交版本(`be65d241e`/`6fc433ea2`),经确认改为快进追加独立
  提交 `e55860f22`,不改写已推送提交
- **批B(注册表解析单源化,`2a239d18a`)**:
  - 删 `backend_supports_layerwise`(生产零调用者),字段存在性只剩
    `get_layerwise_protocol` 一种问法;真值表并入
    `TestLayerwiseProtocolRegistry`(归一化/无协议条目/解析非 None)
  - layout 层删冗余 `strip().lower()`(归一化由解析函数内部承载);
    补 `extract_layout_config` 直接单测(opt-in 真假两路)
  - fixture 构造派生:`_make_gva_worker` /
    `test_partial_prefill_is_saved_and_loaded_for_reused_layer` /
    `test_layerwise_mtp_hit_uses_safe_load_extent` /
    `TestKVPoolSchedulerGetLayerwiseGvaHitTokens._make_scheduler`
    四处,从"事后翻 `use_gva_layerwise`"改为构造传
    `extra_config={"backend": "memcache"}` 由构造函数派生
    (`_layerwise_key_factory` 非 None);其中 hit_tokens 用例保持
    `use_layerwise=False`——其期望值围绕 `query_start_block`
    offset 数学构造,改 True 会把查询起点从
    `min(computed//bs, len)` 变 0,语义被偷偷改掉(首轮服务器 UT
    1 failed 暴露,`expected=4, actual=2`)
- **验证**:服务器(cxy_cann9.1.0,真 torch)全量
  `tests/ut/distributed/ascend_store/` **313 passed**(含 layout
  测试,收集恢复);本地 ruff 0.14.0 check+format 全绿(26 files);
  最终树与服务器验证树逐字节一致后快进推送(`6fc433ea2..2a239d18a`)
- §3.1 代码片段随之过时两处:`backend_supports_layerwise` 已删除
  (单解析器);layout 的 opt-in 判断移入协议
  (`protocol.extract_layout_config`),不再手写 `use_layerwise` 判断

### 4.4 服务器 e2e 验证(2026-09-01,165 / refactor_165,全 PASS)

验证对象 `2a239d18a`(7 commits)+ vllm @ `ba07e4a48`;详细报告与自包含证据在
`../test/`(e2e-report-20260901.md + evidence/),明细对账见
`server-validation.md` 结果记录表:

- **验证项 1(MultiConnector PD 分离,#14465 回归点)PASS**:DSV2-Lite
  P TP=4(MultiConnector[MooncakeLayerwise + AscendStore/memcache
  layerwise])+ D TP=4 + layerwise proxy;请求 5/5 成功(含 2 条
  GSM8K-lite 真实问题);P/D 日志无 AttributeError,
  `_configure_layerwise_reuse_completion` 初始化路径确认走过
- **验证项 2(memcache layerwise 冒烟)PASS**:`hit_check hit_tokens=3456`,
  `load_gvas valid_gvas=27 lease_fail=0`;三维证据链
  alloc=28/stored=28/query=400;external hits=10240
- **验证项 3(mooncake 非 layerwise 零回归)PASS**:Qwen3-32B 请求 100%;
  无 load_gvas/hit_check 标记;master 三维 939.5MB/112 keys/4 clients;
  external hits=10240
- 过程问题均为测试基建修正(子 connector 禁写 engine_id、HBM 残留、
  pipefail grep 静默退出、memcache 1.2.0 指标名实态),被测代码零改动

## 5. 测试覆盖

| 文件 | 覆盖 |
|---|---|
| `test_gva_protocol.py` | gate 真值表(memcache/mooncake/yuanrong/大小写/空白/False 组合);未知 backend 返回 False;memcache 排他性(5 个 GVA 方法仅被 MemcacheBackend override);三类 key 的字节级快照;hit_check 与 full_key 的 rank 格式一致性;`extract_layout_config` opt-in 真假两路 |
| `test_backend.py` | `TestLayerwiseProtocolRegistry`:协议解析非 None、backend 名归一化(MEMCACHE/带空白)、无协议条目返回 None;条目带字段 ⇔ 类 override 5 个 GVA 方法互锁 |
| `test_ascend_store_connector.py` | `set_external_slot_release_waiter` 转发契约:worker gate 拒绝(非 GVA backend)/接受两路;connector 不持有 flag 的回归守卫 |
| `test_pool_worker.py` | worker gate 的接受/拒绝路径;`_make_gva_worker` 等 fixture 经构造派生(`extra_config={"backend": "memcache"}`),派生属性与源一致 |
| `test_layerwise_cache_layout.py` | reuse config 三向:memcache+layerwise 命中(同一 dict 引用)、mooncake 拒绝、memcache 未 opt-in 拒绝 |

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

## 7. 遗留与后续

### 7.1 part2 设计输入:注册表数据绑定 + 协议对象(2026-09-01 定稿)

把四个泄漏点一起解掉的方案是"两件套"——先给设计,再记它与被否能力表的差异。

**四个泄漏点(part2 清零目标)**:

1. **import 泄漏**(布局期/构造期静态依赖):worker / scheduler /
   layout 仍 import `GVAKeyFactory` / 协议模块
2. **运行期分支泄漏**:worker 内 8 处 `self.use_gva_layerwise` 分支
3. **kv_transfer 泄漏**:线程入口 `assert isinstance(m_store,
   MemcacheBackend)` + 顶部 `MemcacheBackend` import
4. **契约泄漏**:base.py 的 5 个 GVA 桩出现在通用 Backend 契约上

**件一:backend_map 条目携带协议模块路径**(解泄漏 1)。
`backend_map` 的 memcache 条目已有 `layerwise_protocol` 字段(提交
`fce5b7807` 落地);通用侧从此不问"是不是 memcache/GVA",改问
"这个 backend 带不带 layerwise 协议",经注册表两个中性解析函数问答。
`use_gva_layerwise` 退化为协议模块内部函数,通用层不再 import。

**件二:协议对象**(解泄漏 2/3/4,行为变更集中区)。
part2 的 GVASession 天然就是这个角色,装配方向:worker / scheduler
构造期一次解析 `self.layerwise = get_layerwise_protocol(...)`(或
None),运行期分支条件从 `if self.use_gva_layerwise` 变为对象存在性
`if self.layerwise is not None`;key 构造、租约、hit-check、
write-finish 全在协议对象内,**方法名按 worker 的需要起
(prepare_save / probe_hits),不按 GVA 线上概念起**。base.py 的 5 个
桩随之删除:GVA store 调用走协议对象(对象内部持 store 引用),不再
出现在通用 Backend 契约上——当初 ABC 方案想做的事,在行为变更的
part2 里做才是合法的。kv_transfer 线程的
`assert isinstance(m_store, MemcacheBackend)` 自然消亡:线程同时拿
`m_store`(通用传输)+ `protocol`(None 或 GVA),GVA 调用走
protocol——断言保护的不变量被类型本身替代。

**终态度量(验收标准)**:`grep -i gva` 在 pool_worker /
pool_scheduler / layerwise_cache_layout / kv_transfer / base.py /
backend/__init__.py 全部零命中;GVA 只存在于 gva_protocol.py、
memcache_backend.py 的装配处、backend_map 的一个字符串值。

**五步落地顺序(part2 内,依赖倒排)**:

| 步 | 内容 | 性质 |
|---|---|---|
| 1 | 注册表加字段 + gva_protocol 补 `extract_layout_config`(先立管道) | 零行为变 |
| 2 | worker/scheduler/layout 三处改走解析(消 import 泄漏) | 零行为变 |
| 3 | 线程改双参数(m_store + protocol),删 isinstance 断言 | 行为变更 |
| 4 | GVA 调用整体进协议对象,删 base.py 5 桩 | 行为变更(大头) |
| 5 | grep 度量验收 | 验收 |

**与被否能力表的三个本质差异**(记录于此,后续评审可能再问):

- 泛化方向:能力表是任意 capability 字符串查询、为"未来第二能力"
  预留;协议字段是单一具名字段,它本身就是反转机制
- 不用它会怎样:能力表不用,通用代码写 `== "memcache"` 无损失;
  协议字段不用,通用代码必须硬编码 memcache 字符串——字段是避免
  泄漏的唯一途径
- 类比:前者是投机扩展点;后者是接口/插件槽,依赖反转都要交这笔税

**代价(已认知)**:一跳间接(调试多经一层 entry→module,靠排他性
测试锁死缓解);中性命名的轻微失真(通用层说 "layerwise protocol"
而今天只有 GVA 一个实现——抽象是为局部性墙与迁移缝,不为扩展);
一次性迁移成本(8 处分支 + import 点 + 线程签名 + base 桩删除)。

### 7.2 决策:步骤 1-2 提前进 part1

2026-09-01 决策:**进**。part1 提前消掉 3 处 import 泄漏、PR 叙事更
完整;步骤 3-4(行为变更)仍严格留在 part2。操作前已备份:
`git branch backup-pre-protocol-bind`(指向 `6fc433ea2`,即 5 提交
head,不含工作区改动)。

### 7.3 当前实施状态(已完成并推送,2026-09-01)

**已闭环**:步骤 1-2 连同测试适配作为提交 6-7 推送,head `2a239d18a`
(7 提交),服务器全量 UT 313 passed;见 §4 末尾"步骤 1-2 提交完成"
条目。以下为当时的实施快照(留档):

| 文件 | 改动 |
|---|---|
| `backend/gva_protocol.py` | 新增 `extract_layout_config(extra_config)`:`use_layerwise` 时返回 extra_config,否则 None(+15) |
| `layerwise_cache_layout.py` | `get_layerwise_reuse_config` 从 `backend_supports_layerwise` + 手写 `use_layerwise` 判断,改为 `get_layerwise_protocol(backend_name)` + `protocol.extract_layout_config(extra_config)`(+17/−6) |
| `pool_worker.py` | 删 `GVAKeyFactory` import;构造期 `self.layerwise_protocol = get_layerwise_protocol(self.backend_name)`、`self.use_gva_layerwise = use_layerwise and self.layerwise_protocol is not None`、`self._layerwise_key_factory`;`_make_layerwise_gva_key` / `_make_layerwise_partial_key` 两处调用点切 `self._layerwise_key_factory`(+17/−7) |
| `pool_scheduler.py` | 同 worker:删 import、构造期三属性、`_make_layerwise_gva_keys_for_hit_check` 切换(+15/−4) |

行为等价性:`get_layerwise_protocol(name) is not None` 与
`backend_supports_layerwise(name)` 对同一字段存在性判定恒等;
`extract_layout_config` 把 `use_layerwise` 判断从 layout 层移入协议
模块,表达式求值结果不变。memcache 之外的 backend(含 mooncake
默认)解析为 None,worker/scheduler UT fixture 的
`"backend": "mooncake"` 路径零 import 变化。

### 7.4 交接清单(三项全部完成,2026-09-01)

三项已闭环:测试适配按"构造派生"落地(四处 fixture,含
hit_tokens 用例保持 `use_layerwise=False` 的 offset 语义);
`backend_supports_layerwise` 已删(单解析器,真值表并入
`TestLayerwiseProtocolRegistry`);验证提交推送完成(313 passed,
head `2a239d18a`)。以下为当时清单(留档):

1. **测试适配(预期必改)**:
   - `test_pool_worker.py` 的 `_make_gva_worker` 等直接赋
     `worker.use_gva_layerwise = True`(fixture 是 mooncake backend),
     改后 `_layerwise_key_factory` 仍为 None——走到
     `_make_layerwise_gva_key` / `_make_layerwise_partial_key` 的用例
     (约 :255/:966/:1243 相关)会 AttributeError,fixture 需同步
     `worker._layerwise_key_factory = GVAKeyFactory`(真实类即可,
     key 是纯字符串构造)
   - `test_pool_scheduler.py:142` 同理(hit_check_keys 路径)
   - `test_gva_protocol.py` / `test_backend.py` 未破坏,可顺手补
     `extract_layout_config` 的直接单测(use_layerwise 真假两路)
2. **待决策:`backend_supports_layerwise` 的去留**。工作区改动后
   生产代码零调用者(三个消费者全切 `get_layerwise_protocol`,
   "支持"由"返回非 None"承载);两个函数并存 = 字段存在性有两种
   问法,又成双源。倾向:删除 + `test_backend.py:28/:90-:102`
   真值表改造(归一化用例并入 `get_layerwise_protocol` 测试)
3. **验证与提交**:
   - 本地 Windows 无 torch:上 192.168.13.165 跑
     (容器 `cxy_cann9.1.0`;`refactor_812` 与新基线不兼容,勿用);
     全量 `tests/ut/distributed/ascend_store/`(test_coordinator 2 例
     为本地 stub 既有失败,CI 有真实 vllm 可过)+ ruff
   - 通过后 `git commit -s` 独立提交(part1 尾巴,行为保持,叙述
     "resolve the layerwise protocol through the registry, key
     construction through the protocol object"),force-push 触发 CI,
     PR 描述补一行 import 泄漏清零
   - 步骤 3-5(线程双参 + 删断言、协议对象 + 删 5 桩、grep 度量)
     全部留给 part2,按 §7.1 顺序实施

### 7.5 原 part2 遗留盘点(步骤 1-2 之外,仍全部有效)

- worker 内 8 处 `self.use_gva_layerwise` 读取(pool_worker.py:381/423/
  472/487/525/830/1167/1333)仍在调用方分支判定,能力知识尚未完全沉入
  backend 接口(件二协议对象的领域)
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

## 8. 检视返工轮(Pz1116 7 条 + gemini 2 条,2026-09-01)

核心立场(用户定调):layerwise 传输路径是通用能力(mooncake 后续复用),
GVA/memcache 专属知识不得出现在通用层;后端专属的只有 key 格式与
opt-in 门禁。方案与 5 个设计决策(D1-D5)见
`../develop_new/layerwise_protocol_refactor_plan.md`。

**4 个 commit,fast-forward 推送(`2a239d18a..359876493`)**:

| 提交 | 内容 |
|---|---|
| `f057d367c` | 协议四函数迁入 `memcache_backend.py`(`make_full_key` / `make_partial_key` / `make_hit_check_keys` / `extract_layout_config`),key 字符串逐字节不变;`gva_protocol.py` 暂留纯别名 |
| `5293cad05` | registry `layerwise_protocol` 字段改布尔标记,resolver 经 `path` import 返回模块(D2);worker/scheduler 改名 `use_layerwise_transfer`、key 构造切协议函数、`._layerwise_key_factory` 删除;测试同步 |
| `5b43d7617` | 删 `gva_protocol.py` 与 `test_gva_protocol.py`;key 快照迁入 `test_backend.py::TestLayerwiseKeyFormats`;新增 `TestLayerwiseProtocolMemcacheExclusivity`(全 backend 协议暴露 ⇔ 布尔标记 ⇔ 类覆写三方一致,负向断言其他 backend 不携带) |
| `359876493` | `kv_transfer.py` 删两处 `isinstance` assert、`MemcacheBackend` import 与 GVA 注释(D4,gemini assert→TypeError 建议随之消解);connector/worker docstring 中性化(D5);waiter 初始化顺序加固 + `test_worker_ready_waiter_handover`;线程 fixture 去 `spec=MemcacheBackend` |

**验证**:refactor_165 内 `git am` 4 补丁后全量
`tests/ut/distributed/ascend_store/` **314 passed**(较 313 +1,新增
waiter 测试);验证树 reset 至 `359876493`,与推送树哈希逐字节一致
(`133fcec2`);本地 ruff 0.14.0 check+format 全绿。
`use_gva_layerwise` / `GVAKeyFactory` / `gva_protocol` 全仓零残留;
`MemcacheBackend` 仅剩 registry 类名字符串与 memcache_backend.py 自身。

**待办(更新 2026-09-01 晚)**:PR 描述已更新(REST PATCH)、回复评论已发
(comment-5491546402);CI 第一轮 mypy 3.10 报 6 错(backend_map 布尔标记使
mypy 把条目 join 成 object),修复为显式注解 `dict[str, dict[str, Any]]`
(原 `5c550766d`)。随后分支 rebase 到 upstream/main `72a988f9d`(带入
#15364 KV Pool 改动),force-with-lease 推送,新 head `63be9e03b`
(12 commits)。**e2e 三场景在新 head 复测全部 PASS**(18:31–18:57,
refactor_165;S2 hit_tokens=3328/valid_gvas=26、S3 三维 939.5MB/112 keys/
4 clients、S1 5/5 无 AttributeError;部署走 git bundle,服务器到 GitHub
间歇断连)。详报 `../test/e2e-report-20260901-rebase.md`,结果记录
`server-validation.md` 复测轮小节。剩余观察:CI 在 rebase 后 head 上的
mypy 3.10/3.11/3.12 结果。

§7.5 遗留盘点中两条已被本轮消掉:"线程类持有 `MemcacheBackend`
import"与"gate 命名/协议对象"相关条目;`GVA` 地址语义命名
(`_allocated_gvas` 等)按 plan §4.3 显式保留。
