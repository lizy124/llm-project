# PR #15367 逐处走读(walkthrough)

> 分工:`PR-15367 record.md` 记录前因、决策演化与 PR 间关系;本文档逐 hunk
> 解释"改了什么、为什么这样写、付出了什么代价"。基审结论:设计成立。
> 代码:`refactor_layerwise_part1` @ `1ff8dc141`(5 commits,基于 `e8f47fc11`)
> 复核基准:每条陈述均可在本地 diff 中逐一指认;快速命令见 §7。

## 0. 全景

```
C1 bfd6f3354  gva_protocol.py 新建:use_gva_layerwise 单点(双参)+ gate/排他性测试(纯新增,无消费者)
C2 6bb71ce78  线程入口断言 + 既有线程测试 spec 适配
C3 e75fc891e  GVAKeyFactory + 消费者切换(worker/scheduler 派生、key、worker gate)
C4 7f3f7e31a  layout 派生切换 + UT stub 同步(实际仅 2 文件,见 §4.3)
C5 1ff8dc141  gate 下沉:connector 纯转发 + 删 #15291 副本
```

依赖方向:C3/C4 依赖 C1(函数已定义);C5 依赖 C3(worker gate 先存在);
C2 只依赖 C1。最终 diff:12 files,+444/−68(`backend/__init__.py` 与
`test_backend.py` 回归基线,不在 diff 面内)。

---

## 1. C1 `bfd6f3354`:GVA 协议模块与单点 gate(gva_protocol.py 新建 + test_gva_protocol.py 新建,+251)

### 1.1 模块选址:为什么是 `backend/gva_protocol.py`

GVA 是 memcache 专属协议——这是本 PR 的领域基石,模块选址由它推出:

- **不进 `MemcacheBackend` 类**:布局期调用点
  `get_gva_layerwise_config()` 手上只有 config 字符串、无任何实例,
  类方案必然要求另一份静态映射兜底(两套机制);且 GVA 的编排
  (session/租约/hit-check)跨 worker/scheduler,塞进 store 客户端
  类违反"协议不持有 worker 状态"(PR-A 评审既定裁定)。
- **不进 `backend/__init__.py`**:领域函数污染包入口(`__init__` 的
  自然读者要看的是包结构,不是 GVA gate);更关键的是 `__init__` 的
  零 import 是**无防护约定**——明天谁加一个 re-export,布局期 import
  安全就悄悄崩了。独立模块的零 import 是**结构性保证**:一个只有
  表和纯函数的模块没有理由 import 任何东西。
- **不进 `memcache_backend.py`**:顶部 `import torch` + 懒加载
  `memcache_hybrid`,协议知识并入会拖重依赖、破坏 UT stub 纯度
  (见 §4.2 的 exec 机制正是靠 `__init__`/`gva_protocol` 零 import
  才成立)。
- **`gva_protocol.py` 与 `GVAKeyFactory` 同住**:memcache 专属的一切
  物理聚拢,文件系统即文档;part2 的 GVASession/GVAHitChecker 归宿
  已定(同模块)。

### 1.2 双参签名(抽象边界的最终裁定)

```python
def use_gva_layerwise(use_layerwise: bool, extra_config: Mapping[str, Any]) -> bool:
    backend_name = str(extra_config.get("backend", "mooncake")).strip().lower()
    return use_layerwise and backend_name == "memcache"
```

- **参数分工是数据流事实**:`use_layerwise` 在 worker/scheduler 是
  **构造参数**(connector 读 config 后传入,worker 另有 10+ 处非 GVA
  layerwise 行为读它),必须由调用方作为权威值传入;layout 则自读
  config,同源。backend 的键名、默认值(`"mooncake"`)、str+strip+
  lower 归一化——这些"backend 字符串怎么读"的知识全部归函数。
  旧标量签名 `(bool, str)` 的问题正是把这部分留在了调用方。
- **为什么不是纯 dict-in(吃整个 extra_config)**:dict-in 会让
  worker 的 gate 从"信构造参数"变成"重读 config",与
  `self.use_layerwise` 形成隐性双源——3 个 UT 失败
  (`_make_worker(use_layerwise=True)` 不写 config)当场证伪了
  这条路。双参是"知识的完整归属"与"数据流权威不搬家"的交点。
- **为什么裸函数而非表 + 查询函数**:能力表是为"未来第二能力"
  预留的泛化机械;专属协议的事实不需要它(见 record §3.1 形态演化)。
- docstring 完整记录 #14465 教训——函数存在的理由就是那段历史。

### 1.3 测试:排他性直连断言

- `test_truth_table`:7 组用例(双参形态),含归一化契约
  `(True, {"backend": "MEMCACHE"}, True)`、`(True, {"backend":
  " Memcache "}, True)` 与键缺失默认 `(True, {}, False)`。
- `test_unknown_backend`:未知名字返回 False。
- `test_gva_store_methods_only_on_memcache_backend`(**核心**):
  遍历 `backend_map`(importlib 动态加载类),对 5 个 GVA 方法检查

  ```python
  owns_override = any(method in vars(cls) for cls in backend_class.__mro__ if cls is not Backend)
  ```

  原理:子类 `vars` 里出现该方法 = 真实 override;继承自 `Backend`
  的桩只存在于 Backend 自己的 vars。断言 `owns_override ==
  (name == "memcache")`——"GVA 是 memcache 专属协议"从 docstring
  升级为可执行断言,新 backend 加入 `backend_map` 即自动纳入检查。
  旧形态的表↔实现互锁(两个事实源对账)随之简化为单事实源直证。

### 1.4 C1 就带测试(与旧形态的差异)

gate 测试随模块在 C1 落地而非推迟到 C2:模块 + 测试是自洽单元,
且 `gva_protocol` 子模块经 UT stub 包 `__path__` 直接解析(见
§4.2),C1 单独 checkout 即可运行测试——旧形态"helpers 住
`__init__.py` 导致 C2/C3 UT 断档"的问题从根上不存在。

---

## 2. C2 `6bb71ce78`:线程入口断言(kv_transfer.py +13,test_kv_transfer.py +13/−4)

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

### 2.4 gate 测试的位置

旧形态把能力测试放 test_backend.py(随 C2);新形态下 gate 与排他性
测试在 C1 随模块落地于 test_gva_protocol.py(见 §1.3)——测试跟知识
走,GVA 的测试物理聚拢,`test_backend.py` 回归基线零 diff。

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

## 3. C3 `e75fc891e`:GVAKeyFactory + 消费者切换(6 文件,+87/−39)

内容最多的提交,实际包含四类改动(key 工厂、partial index 迁移、
**worker/scheduler 派生切换**、worker gate),后两类是切分时混入的,
见 §3.7。

### 3.1 gva_protocol.py:`GVAKeyFactory`(扩展现有模块)

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

### 3.3 pool_scheduler.py

- import:`backend_map`(backend)+ `GVAKeyFactory, use_gva_layerwise`
  (gva_protocol)。
- **派生切换**(此 hunk 在 C3,不是 C4):

  ```python
- self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"
+ self.use_gva_layerwise = use_gva_layerwise(self.use_layerwise, extra_config)
```

  附带把 scheduler 处 `extra_config` 提为局部变量(原是从
  `kv_connector_extra_config` 直接链式取 backend 键)。

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

- `test_gva_protocol.py`(追加快照段):
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

### 3.7 旧形态的切分问题(历史记录,新形态已消除)

1. **派生切换 hunk 混入 C3**(旧形态同样存在,新形态保留此切分,
   且 C3' message 已如实记录):worker/scheduler 两处派生切换在 C3,
   仅 layout 在 C4。
2. **C2/C3 中间态 UT 不可运行(已消除)**:旧形态的 helpers 住
   `backend/__init__.py`,而 UT stub 包的 `__init__` 是手工 mirror,
   C4 之前缺 `_BACKEND_CAPABILITIES`/`use_gva_layerwise`,
   ImportError——"每提交独立可验证"仅在生产行为层面成立。
   **新形态下 gate 函数住 `gva_protocol.py` 子模块,经 stub 包
   `__path__` 直接解析(不依赖 exec),C3' 上用旧 stub 实测
   151 passed**——断档从根上消失,这也是"知识住对地方"的直接
   工程回报。

---

## 4. C4 `7f3f7e31a`:layout 切换 + stub 同步(实际 2 文件,+21/−20)

### 4.1 layerwise_cache_layout.py

```python
- if str(extra_config.get("backend", "mooncake")).lower() == "memcache" and extra_config.get("use_layerwise", False):
+ if use_gva_layerwise(extra_config.get("use_layerwise", False), extra_config):
```

- 第三处(也是最后一处)派生切换。原表达式 `backend == "memcache"
  and use_layerwise` 与新调用仅在 and 操作数顺序上不同,两者皆为纯
  布尔、无短路副作用,等价。
- 该调用点在布局构建期,是"gate 函数必须无实例可查且零重 import"
  的根本原因(见 §1.1/§1.2)。backend 键的读取从此归函数(layout
  处它本就是从 config 读的,同源无争议)。

### 4.2 _mock_deps.py:从手工 mirror 到 exec 真实 `__init__.py`

```python
_backend_init_path = os.path.join(_backend_pkg.__path__[0], "__init__.py")
with open(_backend_init_path, encoding="utf-8") as _backend_init_file:
    exec(compile(_backend_init_file.read(), _backend_init_path, "exec"), vars(_backend_pkg))
```

- **旧机制的问题**:手工 mirror 的 `backend_map` 是 stub 里的第二
  事实源,真实表一改它就漂,mirror 模式注定跟不住。
- **为什么不直接让真实包生效**:stub 包存在的意义是把 ascend_store
  及其 backend 从 torch_npu 等重依赖中隔离(`sys.modules` 已被 stub
  占位,import 机制永远命中 stub);而 `backend/__init__.py` 本身是
  纯 dict、零 import,exec 它不会拉起重依赖——这是"只同步这一个
  文件"的安全边界。
- **为什么用 exec 而非 importlib**:目标路径的 `sys.modules` 条目
  已被 stub 占据,无法再经 import 机制加载真实 `__init__`;exec +
  `vars(_backend_pkg)` 是把真实源码注入 stub 命名空间的最小手段,
  `from ...backend import backend_map` 从此命中真实定义。
- **gva_protocol 为什么不需要特殊处理**:worker/scheduler/layout
  import 的是**子模块**,Python 经 stub 包的 `__path__`(指向真实
  backend 目录)直接加载真实 `gva_protocol.py`——零 import 的纯
  模块,加载无副作用。这正是 §3.7 所说"C3' 上旧 stub 也能跑 UT"
  的机制基础。
- 附带删除了 mirror 版的 `_backend_module_paths` 字典。

### 4.3 commit message 与实际范围的偏差(旧形态历史)

旧形态 C4 message 的 "at all call sites" 以 hunk 论失实
(worker/scheduler 在 C3)。新形态的 C4' message 已改为
"at all call sites" 语义下的收尾描述并如实说明 worker/scheduler
已切换。

---

## 5. C5 `1ff8dc141`:gate 下沉(2 文件,+59/−5)

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
# PR 自身 diff 面(基线 e8f47fc11,应恒为 12 files +444/−68)
git -C $R diff upstream/main HEAD --stat
# 逐提交内容核对
git -C $R show --stat bfd6f3354             # C1:gva_protocol + 测试(2 文件)
git -C $R show --stat e75fc891e             # C3:6 文件,含 worker/scheduler
git -C $R show --stat 7f3f7e31a             # C4:仅 layout + _mock_deps
git -C $R show 7f3f7e31a -- "*pool_worker.py" "*pool_scheduler.py"  # 空 diff
# 全仓派生单点验证(应只剩 3 个消费调用点 + 定义 + docstring/注释)
git -C $R grep -n "use_gva_layerwise" -- "vllm_ascend/**"
# 中间提交可验证性(§3.7:新形态无断档,C3 用旧 stub 应 151 passed)
git -C $R checkout e75fc891e
python d:\lzy\project\kv_pool\run_ascend_store_ut.py tests/ut/distributed/ascend_store/test_gva_protocol.py tests/ut/distributed/ascend_store/test_pool_worker.py tests/ut/distributed/ascend_store/test_pool_scheduler.py tests/ut/distributed/ascend_store/test_kv_transfer.py --noconftest -q -p no:cacheprovider
git -C $R checkout refactor_layerwise_part1
# 修复后全量(2 个 coordinator 失败为本地 stub 既有,非本 PR)
python d:\lzy\project\kv_pool\run_ascend_store_ut.py tests/ut/distributed/ascend_store/ --noconftest -q -p no:cacheprovider --ignore=tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py
```
