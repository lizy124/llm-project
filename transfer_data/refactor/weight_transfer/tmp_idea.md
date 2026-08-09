# weight transfer 重构思路

## 目标

在 `weight_transfer_refactor` 分支上，只重构 `vllm_ascend/distributed/weight_transfer` 目录内的实现，保留现有对外行为和测试预期，同时吸收 `weight_transfer_refactor-pre-rebase-dd7edce97` 中更清晰的设计思想，但不直接复用旧代码实现。

## 现状判断

当前目录里只有 4 个文件：

- `__init__.py`
- `hccl_engine.py`
- `npu_ipc_engine.py`
- `packed_tensor.py`

其中主要问题集中在：

1. `npu_ipc_engine.py` 过于臃肿
   - 同时包含 UUID 生成、生命周期控制、非 packed / packed 发送、HTTP/Ray 发送、IPC handle 合并、worker 侧解包逻辑。
   - 结果是职责交叉，测试与维护都比较难。

2. `hccl_engine.py` 和 `npu_ipc_engine.py` 的模式不统一
   - HCCL 侧已经有 packed broadcast 相关逻辑，但仍然把生命周期、加载策略、传输策略揉在一起。
   - 两个 engine 的“接收端如何加载权重”逻辑应该更清晰地抽成可替换策略。

3. `packed_tensor.py` 里混了三类事情
   - HCCL packed broadcast
   - NPU IPC packed producer / consumer
   - 低层张量打包、拆包与 stream 同步
   - 这让文件既像工具库，又像传输实现。

4. `__init__.py` 只做注册，但注册语义现在偏硬编码
   - 当前通过版本判断决定是否注册 trainer engine。
   - 后续如果再扩展 backend / alias，会更适合做成更明确的注册入口。

## 想保留的思路

参考备份分支里我认为值得保留的方向，但要重新实现：

- 把“生命周期控制”和“加载权重策略”分离
- 把“trainer 侧 handle 收集 / 聚合 / 发送”独立出来
- 把“设备 UUID / 物理设备映射”独立出来
- 把“engine 注册”和“alias 兼容”集中管理
- 让 `npu_ipc_engine.py` 回归为真正的 engine 入口，而不是全家桶

## 建议的目标结构

只在 `vllm_ascend/distributed/weight_transfer` 下重构，不碰别的目录的业务逻辑。

### 1. `device_mapping.py`

职责：
- 解析 `ASCEND_RT_VISIBLE_DEVICES`
- 完成 logical device → physical device 的映射
- 生成稳定的 NPU IPC UUID

收益：
- 避免 `npu_ipc_engine.py` 直接处理环境变量细节
- 逻辑清晰，后续若设备映射规则变化，改动单点即可

### 2. `lifecycle.py`

职责：
- 定义 weight update 生命周期策略接口
- 提供至少两种策略：
  - checkpoint / layerwise reload 风格
  - 直接原地更新风格

收益：
- `start_weight_update` / `finish_weight_update` / `load_weights` 的行为从 engine 中抽出去
- worker 侧可以更清楚地按 backend 选择策略

### 3. `trainer_send.py`

职责：
- trainer 侧的参数遍历与 metadata 收集
- IPC handle 的 all-gather / merge
- send mode 分发（callable / ray / http）

收益：
- `npu_ipc_engine.py` 不再负责发包细节
- HCCL / NPU IPC 可以共用一部分 trainer-side 思路

### 4. `registry.py`

职责：
- 集中注册 Ascend backend
- 管理 `nccl -> hccl`、`ipc -> npu_ipc` 的兼容关系
- 用 lazy loader 避免过早导入可选依赖

收益：
- `__init__.py` 只保留一个很薄的入口
- patch 逻辑也更容易解释和维护

### 5. `packed_tensor.py`

职责：
- 只保留 packed 张量的底层打包 / 解包能力
- 不再承担注册、生命周期、发送协议这些上层职责

建议保留的能力：
- HCCL packed broadcast
- NPU IPC packed producer / consumer

建议调整的方向：
- 把函数语义收紧，减少“一个文件里所有 transfer 变体都堆在一起”的感觉

### 6. `npu_ipc_engine.py`

职责：
- 作为 NPU IPC 的主 engine
- 只负责：更新信息解析、生命周期接入、worker 侧接收、trainer 侧静态入口 / 兼容入口
- 业务细节委托给上面几个辅助模块

我倾向于把它拆成两层：
- worker 侧 engine 类
- trainer 侧 helper / adapter

但不要机械照搬旧分支的类名和方法布局，要根据当前分支的代码风格重新组织。

## 重构原则

1. 先切职责，再切文件
   - 不先追求“文件数越多越好”
   - 先让每个模块只做一类事

2. 维持行为兼容
   - 现有 e2e / ut 测试预期不能坏
   - 对外 API、JSON payload、weight transfer 语义先不改

3. 不直接复制备份分支代码
   - 可以参考旧分支的分层思路
   - 但实现要重新写，避免大段同构代码

4. 优先降低 `npu_ipc_engine.py` 的复杂度
   - 这是当前最值得动的点
   - 其它文件的调整尽量围绕它展开

## 推荐的实施顺序

1. 先抽 `device_mapping.py` 和 `lifecycle.py`
2. 再拆 `trainer_send.py`
3. 然后收敛 `npu_ipc_engine.py`
4. 最后整理 `registry.py` 和 `__init__.py`
5. 回归 `packed_tensor.py`，让其只保留底层 packed 工具

## 需要重点注意的兼容点

- `WeightTransferEngineFactory` 的注册方式
- `NPUIPCWeightTransferEngine` 的初始化签名
- `parse_update_info` 对 `ipc_handles_pickled` 的兼容
- packed / non-packed 两条路径的 payload 结构
- `start_weight_update` / `finish_weight_update` 是否与 worker 状态机一致
- 现有 e2e 测试中对 `/pause`、`/start_weight_update`、`/finish_weight_update`、`/resume` 的调用顺序

## 我对最终形态的倾向

我倾向于保留“一个主 engine + 若干小辅助模块”的方案，而不是拆成很多互相强依赖的类层次。

原因：
- 当前目录本身不大，过度抽象会增加理解成本
- weight transfer 的行为强依赖实际通信流程，保持少量清晰模块更适合后续排查问题
- 兼容性逻辑多，过度 OO 化 反而容易让流程更绕

## 下一步

等开始改代码时，我会按这个方向逐步落地，并在每一步保留可回退的最小改动。
