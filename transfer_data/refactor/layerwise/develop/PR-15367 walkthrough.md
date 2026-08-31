# PR #15367 逐处走读(walkthrough)

> 分工:`PR-15367 record.md` 记录前因、决策演化与 PR 间关系;本文档逐 hunk
> 解释"改了什么、为什么这样写、付出了什么代价"。基审结论:设计成立。
> 代码:`refactor_layerwise_part1` @ `bfeaacb14`(5 commits,基于 `9c3cf949d`)
> 复核基准:每条陈述均可在本地 diff 中逐一指认;快速命令见 §7。

## 0. 全景

```
C1 6334f638e  定义能力表与单点函数(纯新增,无消费者)
C2 f1f96928d  线程入口断言 + 能力表测试 + 既有线程测试 spec 适配
C3 a1f4427a5  GVAKeyFactory + 消费者切换(worker/scheduler 派生、key、worker gate)
C4 5c840cf4c  layout 派生切换 + UT stub 同步(实际仅 2 文件,见 §4.3)
C5 bfeaacb14  gate 下沉:connector 纯转发 + 删 #15291 副本
```

依赖方向:C3/C4 依赖 C1(函数已定义);C5 依赖 C3(worker gate 先存在);
C2 只依赖 C1。最终 diff:14 files,+446/−66。

---

## 1. C1 `6334f638e`:能力注册表(backend/__init__.py,+30)

### 1.1 `_BACKEND_CAPABILITIES` 静态表

```python
_BACKEND_CAPABILITIES: dict[str, frozenset[str]] = {
    "mooncake": frozenset(),
    "memcache": frozenset({"gva_layerwise"}),
    "yuanrong": frozenset(),
}
```

- **键空间与 `backend_map` 对齐**(同名同键),能力知识与注册表知识同址。
- **mooncake/yuanrong 显式列空集而非省略**:表格完整表达"每个 backend
  都被审视过、确认无此能力",省略键与"忘记登记"将不可区分。空集与
  缺失在行为上等价(`backend_supports` 用 `.get(name, frozenset())`),
  显式列出纯粹是给人看的。
- **frozenset**:不可变(注册表无运行时修改场景)、membership O(1)。
- **新 backend 忘登记的失败模式**:静默返回 False → GVA 关闭。可接受,
  因为忘登记的必然是非 GVA backend(只有 memcache 实现 GVA 方法),
  行为正确,只是少了显式声明。

### 1.2 两层函数:`backend_supports` + `use_gva_layerwise`

```python
def backend_supports(backend_name: str, capability: str) -> bool: ...
def use_gva_layerwise(use_layerwise: bool, backend_name: str) -> bool: ...
```

- 分工:`backend_supports` 是**通用机制**(表怎么查),`use_gva_layerwise`
  是**领域用途**(当前唯一的派生)。未来出现第二个能力时只加领域函数,
  机制不动。
- `use_gva_layerwise` 的 docstring 完整记录了 #14465 教训——这个函数
  存在的理由就是那段历史,写在定义处让删它的人先读到后果。
- **大小写契约**:函数不做 `.lower()`,三个调用方各自先 lower 再传入。
  这是与原 5 份副本行为一致的刻意选择(原副本也是先 lower 再比较),
  避免本 PR 引入任何行为差。代价:`use_gva_layerwise(True, "MEMCACHE")`
  返回 False,靠调用方纪律保证——真值表测试用 `(True, "MEMCACHE", False)`
  用例把这个陷阱文档化了。更稳的做法(函数内归一化)属于行为变更,
  不在本 PR 范围。

### 1.3 被否的类属性方案

`Backend.GVA_LAYERWISE_CAPABLE = True` 只覆盖 worker/scheduler 两处
(构造后有实例可查);布局期调用点 `get_gva_layerwise_config()` 手上
只有 config 字符串、无任何实例,仍需静态表 → 变成两套机制,弃。
布局期无实例是根本约束;"布局期不能 import torch"不成立(布局代码
本身就 import vllm.config,torch 早已就绪),不作为论据。

### 1.4 为什么 C1 不带测试

C1 是纯新增、零消费者——表此刻不可能被读错(没有读取方)。测试
(真值表 + 表↔实现一致性)与 C2 的断言是一个语义单元,随 C2 落地。

---

## 2. C2 `f1f96928d`:线程入口断言(kv_transfer.py +13,test_backend.py +63,test_kv_transfer.py +13/−4)

### 2.1 顶部 import `MemcacheBackend`(运行时,非 TYPE_CHECKING)

`assert isinstance` 需要运行时类对象,TYPE_CHECKING + 字符串注解无法
满足。**代价**:通用传输层文件(kv_transfer.py)从"只 import 抽象
`Backend`"变为直接依赖具体 backend——在"把 GVA 知识移出通用层"的
收敛方向上是反向一步。接受它是因为这是 part2 线程搬迁前的临时脚手架:
断言随线程类一起搬进 backend 侧后,这个 import 自然消失(已记入
record §7 遗留)。

### 2.2 assert 的位置:`_handle_request` 入口,而非 `__init__`

选入口而非构造期,是因为这个 assert 有**双重职责**:

1. 运行时 fail-fast(构造期也做得到,且更早);
2. **为 mypy 提供方法体内类型收窄**——assert 之后,`self.m_store` 在
   该方法内被静态类型化为 `MemcacheBackend`,后续 5 个 GVA 方法调用
   全部通过类型检查(commit message: "the two call sites are
   statically typed to MemcacheBackend")。放 `__init__` 的 assert
   无法跨方法生效。

这直接回应 #14465 型回归的痛点:该类回归静态不可见、运行时才炸。
有了收窄,mypy 在编译期就能挡住"对 `Backend` 调 GVA 方法"的误用。

### 2.3 为什么是 `assert` 而非 `if not ...: raise`

语义上是**不变量**(线程只在 gate 开启时启动,assert 恒真)而非输入
校验。`python -O` 会剥掉 assert——届时失去提前失败,但 base 桩的
`NotImplementedError` 仍在首次调用处兜底,不产生错误结果。零开销
(单次 isinstance)换错误定位从"某方法缺实现"提前为"gate 不变量
被破坏",这笔账成立。

### 2.4 test_backend.py 的能力测试(63 行,`TestBackendCapabilities`)

- `_GVA_STORE_METHODS`:5 个方法名,与 base.py 的 5 个桩一一对应,
  是"能力"的操作化定义。
- `test_capability_table_matches_gva_store_methods`(核心):遍历
  `backend_map` 全部条目(经 `importlib.import_module(entry["path"])`
  动态加载类),对每个方法检查

  ```python
  owns_override = any(method in vars(cls) for cls in backend_class.__mro__ if cls is not Backend)
  ```

  原理:子类/中间类 `__dict__`(即 `vars`)里出现该方法 = 真实
  override;继承自 `Backend` 的桩只存在于 Backend 自己的 vars。
  断言 `owns_override == 表声明`,把"静态表是第二事实源"的漂移风险
  锁死。新 backend 加入 backend_map 即自动纳入检查。
- `test_use_gva_layerwise_truth_table`:5 组用例,含大小写陷阱
  `(True, "MEMCACHE", False)`。
- 未知 backend / 未知 capability 返回 False:防御性默认的显式锁定。

### 2.5 test_kv_transfer.py 的 spec 适配(首轮 CI 失败的修复)

首轮 CI(head `735065fe1`)7 failed:既有 7 个 layerwise 线程测试
直接调 `_handle_request`,其 fixture 用裸 `MagicMock()` 当 store,
过不了新断言。教训:入口加 isinstance 断言时,必须排查所有直接
调用该入口的既有测试。

修法(`TestGVALayerTransferFailures._make_sending_thread` /
`TestGVALayerReceivingTaskOwnership._make_thread`):

```python
store = MagicMock(spec=MemcacheBackend)
store.store = MagicMock(batch_copy=MagicMock(return_value=0))
```

- **spec=MemcacheBackend**:`MagicMock(spec=X)` 的 isinstance 检查
  通过(mock 的 `__class__` 呈现为 spec 类),与 PR-A b89884b 的
  `spec=_DualSpecStore` 同一模式。线程代码用到的 `set_device` /
  `exists` / `get` / `put` / `batch_write_finish` /
  `batch_remove_lease` 都是类方法,在 spec 的 dir 内,直接可访问。
- **`.store` 必须显式装配**:它是 `__init__` 里赋值的**实例属性**,
  不在 `dir(MemcacheBackend)` 中,spec 模式下 getattr 受限——先
  setattr 一个 child mock(带 `batch_copy`),后续读取命中属性
  字典,不再走 spec 检查。`store.store.batch_copy.return_value=0`
  的旧写法第一句就是受限 getattr,会直接 AttributeError。
- 修复 amend 进 C2:断言属 C2,既有测试的适配同属 C2,单独
  checkout 时代码与测试自洽。

---

## 3. C3 `a1f4427a5`:GVAKeyFactory + 消费者切换(6 文件,+245/−38)

内容最多的提交,实际包含四类改动(key 工厂、partial index 迁移、
**worker/scheduler 派生切换**、worker gate),后两类是切分时混入的,
见 §3.7。

### 3.1 gva_protocol.py:`GVAKeyFactory`(新文件,77 行)

- **为什么是 class + 3 staticmethod 而非模块级函数**:同族字符串构造
  归入一个命名空间;part2 的 GVASession/GVAHitChecker(原 PR-A 设计)
  也规划进该模块,类是既定的组织形式。staticmethod = 纯函数,无状态。
- **字节级兼容是硬约束**:key 是跨集群的线上数据格式(worker 写入、
  scheduler 命中检查、部署集群中已存在的存量数据),漂移一个字符 =
  命中变未命中。`full_key` 单组保持 PR #11585 的 `model@hash@rank`,
  多组 `model@group@hash@rank`;`partial_key` 的 7 段式不变。
- `hit_check_keys` 生成每 rank 一把 key(`range(num_ranks)`),
  rank 数由调用方算好传入(scheduler 侧 `tp_size // put_step`,
  put_step 分组内 MLA 共享一把 key)。

### 3.2 metadata.py:`get_partial_block_index`(worker 静态方法 → 模块函数)

- 函数体逐字迁移,仅去掉 `@staticmethod` 与 `self._` 前缀。
- **为什么迁 metadata.py 而非 gva_protocol.py**:它做的是纯 token/block
  算术(`divmod` + 边界判断),与 GVA 协议无关,且 4 个调用点中
  有非 GVA 的 layerwise 路径——它属于元数据计算域,不属于协议域。
  这也是此前 p0 方案的既定裁定。
- worker 里删除原静态方法,4 处调用从 `self._get_partial_block_index`
  改为直接调模块函数。

### 3.3 pool_scheduler.py(17 行)

- import:`backend_map, use_gva_layerwise` + `GVAKeyFactory`。
- **派生切换**(此 hunk 在 C3,不是 C4):

  ```python
  - self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"
  + self.use_gva_layerwise = use_gva_layerwise(self.use_layerwise, self.backend_name)
  ```

- `_make_layerwise_gva_keys_for_hit_check` 内联的两套 f-string 分支
  整体替换为一次 `GVAKeyFactory.hit_check_keys(...)` 调用;原
  `len(self.kv_cache_group_ids) > 1` 的分支判断内化为工厂的
  `num_groups` 参数。

### 3.4 pool_worker.py(62 行)

- import 同上 + `get_partial_block_index`。
- **派生切换**(同 3.3,此 hunk 在 C3)。
- 4 处 `get_partial_block_index` 调用替换 + 删除原静态方法。
- `_make_layerwise_gva_key` / `_make_layerwise_partial_key` **保留为
  薄包装**(只把 `self.model_name` / `self.head_or_tp_rank` /
  `self.num_kv_cache_groups` 绑进工厂调用)。为什么不删掉包装直呼
  工厂:调用点保持零改动,diff 最小、行为保持最易审。工厂吃显式
  参数(可独立测试),包装绑实例上下文(调用方方便),两层各取所需。

### 3.5 `set_external_slot_release_waiter` 加 gate + 返回 bool

```python
+ if not self.use_gva_layerwise:
+     return False
  self.external_slot_release_waiter = waiter
+ return True
```

签名 `-> None` 改 `-> bool`。**为什么落在本提交**:worker 侧 gate
是"数据面消费者自判"的载体,C3 引入它、C5 才让 connector 消费它。
本提交时刻该方法的唯一调用方(connector)仍持有 #15291 的 connector
侧 gate 且忽略返回值——两道 gate 并存且判定同一表达式,worker 侧
返回值无人消费,行为严格不变。这是"gate 迁移"被拆成两半中安全的那
一半先落地。(此 hunk 原 commit message 漏记,已 amend 补记。)

### 3.6 测试

- `test_gva_protocol.py`(新,83 行):
  - 快照期望值**从重构前的 pool_worker/pool_scheduler 实现转录**,
    不是从新实现生成——否则是同义反复,漂移测不出来。
  - `test_full_key_and_hit_check_key_share_rank_format`:断言
    `hit_check_keys(...)[r] == full_key(..., r, ...)`,即"rank r 的
    命中检查 key 恰是 rank r 的写入 key"。这是防两份构造漂移的
    **本质断言**(比快照更根本:快照锁形状,它锁关系)。
  - 边界:`num_ranks=0 → []`。
- `test_pool_worker.py`:
  - `test_partial_prefill_block_index_boundaries` 等价改写为直接调
    模块函数(原经 `cls._get_partial_block_index`)。
  - 新增 `test_set_external_slot_release_waiter_gated_on_gva`:直接
    设 `worker.use_gva_layerwise` 两态,断言 False 拒绝(不落
    waiter)/ True 接受(落 waiter)。

### 3.7 本提交的切分问题(如实记录,复核时发现)

1. **派生切换 hunk 混入 C3**:record 原表格把"三处派生切换"记在 C4,
   实际 worker/scheduler 两处在 C3、仅 layout 在 C4(C4 对这两个文件
   的 diff 为空)。record 表格已按事实修正。
2. **C2/C3 中间态 UT 不可运行**:UT 的 stub 包(`_mock_deps.py`)在
   C4 之前是手工 mirror,只有 `backend_map`,没有
   `_BACKEND_CAPABILITIES` / `use_gva_layerwise`。于是:
   - C2 单独 checkout:`test_backend.py` 的
     `from ...backend import _BACKEND_CAPABILITIES` 在 stub 下
     ImportError——**C2 引入的测试自己要到 C4 才能跑**;
   - C3 单独 checkout:上者 + worker/scheduler 真实模块 import
     `use_gva_layerwise`,`test_pool_worker/test_pool_scheduler`
     同样 ImportError。
   - 影响面:生产运行行为仍逐提交保持(stub 只存在于 UT);PR CI
     只测 head(全绿)。受影响的是"每个提交单独 checkout 可验证"
     的主张与未来 bisect:在 C2/C3 上跑 UT 会看到与本 PR 无关的红。
   - 理想切分:stub 的 exec 同步(现 C4 的 `_mock_deps` hunk)应随
     C1 或 C2 落地。是否重排历史由作者定夺(需再 force-push)。

---

## 4. C4 `5c840cf4c`:layout 切换 + stub 同步(实际 2 文件,+23/−19)

### 4.1 layerwise_cache_layout.py

```python
- if str(extra_config.get("backend", "mooncake")).lower() == "memcache" and extra_config.get("use_layerwise", False):
+ if use_gva_layerwise(
+     extra_config.get("use_layerwise", False),
+     str(extra_config.get("backend", "mooncake")).lower(),
+ ):
```

- 第三处(也是最后一处)派生切换。原表达式 `backend == "memcache"
  and use_layerwise` 与新调用仅在 and 操作数顺序上不同,两者皆为纯
  布尔、无短路副作用,等价。
- 该调用点在布局构建期,是"静态表必须字符串进布尔出"的根本原因
  (见 §1.3)。

### 4.2 _mock_deps.py:从手工 mirror 到 exec 真实 `__init__.py`

```python
_backend_init_path = os.path.join(_backend_pkg.__path__[0], "__init__.py")
with open(_backend_init_path, encoding="utf-8") as _backend_init_file:
    exec(compile(_backend_init_file.read(), _backend_init_path, "exec"), vars(_backend_pkg))
```

- **旧机制的问题**:手工 mirror 的 `backend_map` 是 stub 里的第二
  事实源,真实表一改它就漂——本 PR 恰恰要往真实 `__init__` 里加
  三个符号,mirror 模式注定跟不住。
- **为什么不直接让真实包生效**:stub 包存在的意义是把 ascend_store
  及其 backend 从 torch_npu 等重依赖中隔离(`sys.modules` 已被 stub
  占位,import 机制永远命中 stub);而 `backend/__init__.py` 本身是
  纯 dict + def、零 import,exec 它不会拉起重依赖——这是"只同步
  这一个文件"的安全边界。
- **为什么用 exec 而非 importlib**:目标路径的 `sys.modules` 条目
  已被 stub 占据,无法再经 import 机制加载真实 `__init__`;exec +
  `vars(_backend_pkg)` 是把真实源码注入 stub 命名空间的最小手段,
  `from ...backend import use_gva_layerwise` 从此命中真实定义。
- 附带删除了 mirror 版的 `_backend_module_paths` 字典。

### 4.3 commit message 与实际范围的偏差

message 主体写 "Switch the consumer-side duplicated derivations
(KVPoolWorker, KVPoolScheduler, get_gva_layerwise_config)"——但
worker/scheduler 的切换实际发生在 C3(§3.7),本提交只完成 layout
一处(可辩护为"完成 all call sites 的收尾",但 hunk 层面失实)。
同时本提交的 `_mock_deps` hunk 实质是在偿还 C2/C3 的 UT 断档(§3.7
第 2 条)。是否 amend 由作者定夺。

---

## 5. C5 `bfeaacb14`:gate 下沉(2 文件,+59/−5)

### 5.1 删除 connector `__init__` 的 2 行——为什么是有意为之

删的这两行是 #15291 刚合入 main 的热修:

```python
- backend_name = str(extra_config.get("backend", "mooncake")).lower()
- self.use_gva_layerwise = self.use_layerwise and backend_name == "memcache"
```

**删除成立的三段论证**:

1. **删的是副本,不是修复本体**。#14465 的 bug 是"读者
   (`set_external_slot_release_waiter`)无副本可读 → MultiConnector
   初始化 AttributeError";#15291 的修法是"在 connector 恢复一份
   副本"。本 PR 的修法是把 gate 移到读者身边(worker)——副本存在的
   前提(读者在 connector)消失,保留它只会重演历史。
2. **结构性免疫,而非又打一层补丁**。删除后 connector 对
   `use_gva_layerwise` 的代码引用为零(仅 docstring 提及设计约束):
   未来的死代码清理无论怎么删 connector 都不可能再删出 #14465;
   而若有人删 worker 侧的 `use_gva_layerwise` 单点,所有消费者在
   **import 期**即失败——错误的可见性从"运行时 AttributeError"
   提前到"import 失败",这类回归从结构上关闭。
3. **取代关系已声明**。C5 的 commit message 记录完整因果链
   (#14465 → #15291 → 取代);PR 描述补一行
   "Supersedes the connector-side flag restored by #15291 (gate
   moved to the worker)",让 reviewer 第一眼知道这 2 行删除的来龙,
   同时在 #15291 页面留下 backlink,闭环其生命周期。

### 5.2 纯转发的实现细节

```python
def set_external_slot_release_waiter(self, waiter: Callable[[int], None]) -> bool:
    if getattr(self, "connector_worker", None) is None:
        return False
    return self.connector_worker.set_external_slot_release_waiter(waiter)
```

- **保留 None 防御**:SCHEDULER role 下 `__init__` 不创建
  `connector_worker`,纯转发必须先挡这一路(沿用了 #15291 前既有
  的 `getattr(..., None)` 写法)。
- **docstring 即路标**:向未来改这段的人说明"connector 不许持有
  派生"及原因——这是用注释固化架构约束,防止下一个 #15291 式的
  "顺手修复"把副本加回来。

### 5.3 bool 协议与 MultiConnector 消费端

- worker 返回严格 bool:True = waiter 已接受;False = 非 GVA 模式
  拒绝。
- 消费端([ascend_multi_connector.py:52](file:///d:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py#L52)):

  ```python
  if callable(set_waiter) and set_waiter(self._wait_for_external_slot_release) is not False:
  ```

  `is not False` 是 MultiConnector 的既有写法(main 上就有),兼容
  返回 None 的旧签名实现;worker 侧精确化后,该表达式对 AscendStore
  子 connector 的判定从"非 False 即配置成功"变为严格布尔,语义只
  收紧不错判。
- **时序安全**:MultiConnector `__init__` 先经 super() 创建全部子
  connector(每个 AscendStoreConnector 创建自己的 worker),之后才
  调 `_configure_layerwise_reuse_completion` → `set_waiter`,转发时
  `connector_worker` 必然就绪。#14465 崩的正是这条链上的属性缺失。

### 5.4 两个测试(回归守卫)

- `test_set_external_slot_release_waiter_worker_gates`:配置
  `use_layerwise=True, backend=mooncake`(即"layerwise 开但非 GVA"
  的组合),mock worker 返回 False/True 两路,断言:connector 原样
  转发返回值、waiter 透传不变。docstring 明确写了 #14465/#15291
  的历史——若有人再把派生塞回 connector,这里的注释与断言会提示
  正确位置。
- `test_set_external_slot_release_waiter_scheduler_role`:SCHEDULER
  role 下无 worker,返回 False(覆盖 §5.2 的 None 防御路径)。

---

## 6. 横切论证

### 6.1 行为等价链(逐处替换的等价依据)

| 替换点 | 等价依据 |
|---|---|
| 三处派生 → 单点函数 | 同一表达式 `use_layerwise and backend=="memcache"`,操作数皆为纯布尔 |
| key 构造 → 工厂 | f-string 逐字相同(快照测试从旧实现转录期望值锁死) |
| layout 的 and 交换 | 两操作数纯布尔,交换无短路副作用 |
| worker gate + bool | C3 时刻唯一调用方忽略返回值;C5 后 gate 表达式与原 connector 侧一致 |
| connector 纯转发 | 转发后判定表达式 = 原 connector 侧判定(worker 持同一派生) |

### 6.2 提交排序的依赖逻辑

定义(C1)→ 断言/测试(C2)→ 消费者切换(C3)→ 收尾切换 + stub(C4)
→ 删除旧 gate(C5)。每步都在前一步成立的前提下做最小增量;C5 放
最后是因为它删除的副本在 C3/C4 期间仍是 connector 的活跃 gate
(#15291 语义),提前删会出现"两道 gate 都不在"的窗口。

### 6.3 已知债务清单(合入后跟踪)

1. **kv_transfer.py 直接 import MemcacheBackend**(§2.1):part2 搬
   线程类时随 assert 一起消除。
2. **大小写契约靠调用方 lower**(§1.2):归一化属行为变更,如做需
   单独提交 + 真值表用例翻转。
3. **C2/C3 中间态 UT 断档**(§3.7):stub exec 若挪到 C1/C2 可修;
   亦可选不动历史、在 record 留档(现状)。
4. **C3/C4 的 message 与 hunk 不完全对应**(§3.7/§4.3):C3 已 amend
   补记 gate hunk,派生切换仍未在 C3 message 中提及;C4 的
   "at all call sites" 以 hunk 论失实。再 amend 需重 force-push。

---

## 7. 快速复核命令

```powershell
$R = "d:\lzy\project\kv_pool\code\vllm-ascend"
# 最终树与首轮 CI 失败 head 的差异仅测试修复(spec 适配)
git -C $R diff 735065fe1 bfeaacb14          # 应只有 test_kv_transfer.py
# 逐提交内容核对(§3.7/§4.3 的依据)
git -C $R show --stat a1f4427a5             # C3:6 文件,含 worker/scheduler
git -C $R show --stat 5c840cf4c             # C4:仅 layout + _mock_deps
git -C $R show 5c840cf4c -- "*pool_worker.py" "*pool_scheduler.py"  # 空 diff
# 全仓派生单点验证(应只剩 3 个消费调用点 + 定义 + docstring/注释)
git -C $R grep -n "use_gva_layerwise" -- "vllm_ascend/**"
# C2/C3 中间态 UT 断档复现(§3.7)
git -C $R stash list; git -C $R checkout 6334f638e
python d:\lzy\project\kv_pool\run_ascend_store_ut.py tests/ut/distributed/ascend_store/test_backend.py --noconftest -q -p no:cacheprovider
# 预期:ImportError(_BACKEND_CAPABILITIES);checkout a1f4427a5 同理多两个文件挂
git -C $R checkout refactor_layerwise_part1
# 修复后全量(2 个 coordinator 失败为本地 stub 既有,非本 PR)
python d:\lzy\project\kv_pool\run_ascend_store_ut.py tests/ut/distributed/ascend_store/ --noconftest -q -p no:cacheprovider --ignore=tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py
```
