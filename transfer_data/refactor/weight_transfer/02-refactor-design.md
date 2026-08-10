# vllm-ascend weight_transfer 重构设计

## 1. 背景

vllm-ascend 的 `weight_transfer` 不是一套独立协议，而是将 upstream vLLM 的 weight transfer 能力适配到 Ascend NPU：

```text
upstream NCCL    -> Ascend HCCL
upstream CUDA IPC -> Ascend NPU IPC
```

上层继续复用 vLLM 提供的配置、worker lifecycle、HTTP API 和 trainer 调用链。重构目标不是重新设计 weight transfer 协议，也不是单纯减少代码行数，而是：

1. 明确 upstream 逻辑与 Ascend 差异的边界。
2. 集中 upstream API 和版本兼容点。
3. 保持 HCCL、NPU IPC、packed/unpacked 现有行为。
4. 让后续对齐 upstream 时可以快速判断需要同步哪些模块。

## 2. 当前基线

### 2.1 分支状态

设计审查时，`weight_transfer_refactor` 与 `upstream/main` 指向同一提交：

```text
ac19e1e64 [BugFix][MRV2] Unify update_stream across main and draft...
```

当前 tracked 源码结构为：

```text
vllm_ascend/distributed/weight_transfer/
  __init__.py
  hccl_engine.py
  npu_ipc_engine.py
  packed_tensor.py

vllm_ascend/patch/platform/
  patch_weight_transfer_engine.py
```

本设计中的 `registry.py`、`lifecycle.py`、`trainer_send.py`、`device_mapping.py` 等均为目标结构，当前尚未实现。目录中的历史 `.pyc` 文件不属于源码，也不作为实现或测试已完成的依据。

### 2.2 upstream 结构

upstream vLLM 的 weight transfer 目录随版本演进，当前可包含：

```text
vllm/distributed/weight_transfer/
  base.py
  factory.py
  nccl_common.py
  nccl_engine.py
  ipc_engine.py
  packed_tensor.py
  sparse_nccl_engine.py
```

vllm-ascend 不应 fork upstream 的 `base.py` 和 `factory.py`。需要通过注册、adapter 和 backend 实现接入 upstream 扩展点。

### 2.3 当前实现特点

当前实现大体可以理解为：

```text
copy upstream implementation
  -> port CUDA/NCCL primitives to NPU/HCCL
  -> patch factory registration
  -> 为不同 vLLM API 保留兼容分支
```

主要问题不是重复代码本身，而是以下内容混在 backend 文件中：

- upstream API 兼容。
- worker lifecycle。
- trainer-side 编排。
- HTTP、Ray、callable 分发。
- packed/unpacked 协议。
- HCCL 通信和 NPU IPC handle。
- logical/physical NPU 映射和节点身份。

## 3. 版本兼容边界

当前代码至少需要面对两类 trainer API。重构前必须根据项目实际支持范围确认版本矩阵，不能假设 upstream 始终只有一套接口。

| vLLM API 形态 | worker 侧 | trainer 侧 | Ascend 适配方式 |
|---|---|---|---|
| legacy，例如 0.26.0 路径 | `WeightTransferEngine` | static `trainer_send_weights(...)` | legacy trainer adapter |
| newer/main 路径 | `WeightTransferEngine` | `TrainerWeightTransferEngine` / `WeightTransferTrainerFactory` | stateful trainer adapter |

版本兼容原则：

1. 版本判断只能出现在兼容入口，不能继续扩散到 transport loop。
2. 公共模块不得在不支持的 vLLM 版本上顶层 import 不存在的类。
3. worker engine 与 trainer engine 分别注册，不能假设两个 factory 总是同时存在。
4. legacy 和 newer 路径必须使用相同的 wire contract，但允许使用不同的调用 adapter。
5. 支持版本范围应以项目发布策略为准；范围确定后，为每一种 API 形态保留最少一个 UT。

建议建立内部兼容模块：

```text
compat.py
  - 探测 upstream trainer API 能力
  - 暴露稳定的类型和注册函数
  - 隔离必要的版本条件分支
```

优先使用能力探测，例如 `hasattr` 或可控的 import 检查；只有 upstream API 无法通过能力区分时才使用版本判断。

## 4. 重构原则

### 4.1 保留 upstream ownership

以下职责继续由 upstream vLLM 所有：

- `WeightTransferConfig`。
- `WeightTransferEngine` 基类及 worker 调用入口。
- upstream factory。
- HTTP endpoint schema 和 client protocol。
- model weight loading 的公共语义。

Ascend 只提供：

- backend 注册和 alias 替换。
- upstream 版本 adapter。
- HCCL communicator 和 broadcast。
- NPU IPC handle 创建、汇聚和 rebuild。
- Ascend packed transport。
- Ascend 设备映射和 IPC identity。

### 4.2 不合并两个 backend

HCCL 是 collective broadcast，NPU IPC 是同机共享内存 handle。两者的同步、资源生命周期和错误模型不同，不应合并成一个 engine，也不应在公共层堆积 `if backend == ...`。

### 4.3 只抽取稳定公共逻辑

适合抽取：

- 参数 metadata 的构造与验证。
- 公共 update record。
- 生命周期状态校验。
- HTTP 基础请求与错误处理。
- 注册和 alias 策略。

不适合强行抽取：

- HCCL broadcast loop。
- IPC handle 编码、all-gather 和 rebuild。
- backend-specific packed producer/consumer。
- backend-specific payload 序列化。
- backend-specific post-send synchronization。

## 5. 目标结构

推荐目标结构：

```text
vllm_ascend/distributed/weight_transfer/
  __init__.py
  compat.py
  registry.py
  lifecycle.py
  metadata.py
  device_mapping.py
  process_identity.py
  hccl_engine.py
  hccl_sender.py
  npu_ipc_engine.py
  npu_ipc_sender.py
  packed_tensor.py
```

是否单独创建 `hccl_sender.py` 和 `npu_ipc_sender.py` 由拆分后的文件规模决定；如果 sender 很短，可继续放在对应 engine 文件中，不为目录结构而拆文件。

职责定义：

- `compat.py`：upstream API 能力探测和版本 adapter。
- `registry.py`：唯一 backend 注册、覆盖和 alias 入口。
- `lifecycle.py`：状态校验及可组合 lifecycle hook，不实现 transport。
- `metadata.py`：name、shape、dtype、tensor size 等稳定数据结构和校验。
- `device_mapping.py`：logical NPU 与 physical NPU 的映射。
- `process_identity.py`：NPU IPC 使用的 host/device identity。
- `hccl_engine.py`：worker 侧 HCCL 初始化、receive 和 shutdown。
- `hccl_sender.py`：trainer 侧 HCCL group、broadcast 和同步。
- `npu_ipc_engine.py`：worker 侧 IPC handle decode/rebuild 和 load。
- `npu_ipc_sender.py`：trainer 侧 handle 创建、汇聚、payload 构造和同步。
- `packed_tensor.py`：Ascend packed 协议及 backend-specific producer/consumer。

示例和测试 helper：

```text
examples/rl/weight_transfer_http_utils.py
tests/e2e/pull_request/weight_transfer_utils.py
```

helper 只能抽公共 HTTP primitive 和 lifecycle driver；HCCL 与 NPU IPC 的 payload encoder、阻塞方式和传输参数仍由 backend-specific 调用方提供。

## 6. Registry 设计

### 6.1 必须支持的名称

```text
hccl    -> HCCLWeightTransferEngine
npu_ipc -> NPUIPCWeightTransferEngine
nccl    -> HCCLWeightTransferEngine
ipc     -> NPUIPCWeightTransferEngine
```

其中 `hccl`、`npu_ipc` 是 Ascend native name；`nccl`、`ipc` 是 upstream-compatible alias。

### 6.2 注册约束

upstream `register_engine()` 对重复名称可能抛出 `ValueError`，而 `nccl`、`ipc` 通常已由 upstream 注册。因此 Ascend registry 必须区分：

```text
register native backend
replace upstream alias
```

禁止让 `__init__.py` 和 platform patch 分别修改同一个 registry。目标调用链应为：

```text
plugin/platform initialization
  -> register_ascend_weight_transfer_backends()
       -> 注册 hccl/npu_ipc
       -> 替换 nccl/ipc alias
       -> 按能力注册 trainer engine
```

注册函数必须满足：

- 同一目标重复调用时幂等。
- 已被无关第三方覆盖时 fail closed，给出明确错误或 warning，不能静默抢占。
- loader 保持 lazy import，不在注册阶段 import Ray、torch_npu 或初始化通信资源。
- UT 中可以保存并恢复 upstream registry，避免测试间污染。

`patch_weight_transfer_engine.py` 最终只负责触发统一注册函数；不再直接 import engine class 或直接写 `_registry`。

## 7. Lifecycle 设计

### 7.1 状态机

生命周期不能只抽成两个无状态 helper。至少需要定义以下逻辑状态：

```text
NEW -> INITIALIZED -> UPDATING -> READY
                         |
                         -> FAILED

任意可终止状态 -> SHUTDOWN
```

允许操作：

- `init_transfer_engine`：`NEW`，根据 upstream 约束决定是否允许重复幂等调用。
- `start_weight_update`：`INITIALIZED` 或 `READY`。
- `update_weights`：仅 `UPDATING`。
- `finish_weight_update`：仅 `UPDATING`。
- `shutdown`：除 `SHUTDOWN` 外的可清理状态，重复调用应安全。

是否在 Ascend 层显式保存完整状态，要结合 upstream 是否已经提供状态管理决定。即使不新增状态字段，也必须通过测试固定非法调用的行为。

### 7.2 Hook 边界

`lifecycle.py` 只能提供可组合 hook 或状态校验，具体 engine 决定：

- 是否需要 `initialize_layerwise_reload`。
- 是否需要 `finalize_layerwise_reload`。
- 是否为 no-op。
- finish/abort 时清理哪些 backend 资源。

lifecycle 选择可能受 backend、upstream API 能力、model loader 和目标 model 影响，不能只按 backend name 分支。

必须考虑：

- `set_weight_update_target()` 后 hook 操作当前 target，而不是默认 model。
- `update_weights` 异常时进入何种状态。
- `finish_weight_update` 异常后能否重试。
- client 失败时是否调用 abort/cleanup。
- `torch.accelerator.synchronize()` 的 ownership，避免重复同步或遗漏同步。

## 8. Trainer-side 设计

### 8.1 公共层

公共层只负责：

1. 从 weight source 迭代参数。
2. 构造和验证稳定 metadata。
3. 驱动 `start -> send chunk(s) -> finish`。
4. 在异常时执行已定义的 cleanup/abort 策略。

建议的稳定接口形态：

```text
WeightMetadata
  - name
  - shape
  - dtype_name
  - tensor_size（packed 时）

BackendSender
  - initialize(...)
  - send(source)
  - post_send_sync()
  - shutdown()
```

不要求创建上述名称的抽象基类；只有当它能消除真实重复且不引入 backend 条件分支时才落地。

### 8.2 Backend-specific 层

HCCL sender 负责：

- rank/world size 和 process group。
- broadcast 顺序。
- packed HCCL producer。
- HCCL 同步和 communicator cleanup。

NPU IPC sender 负责：

- IPC handle 创建。
- rank 间 handle 汇聚。
- physical NPU identity 绑定。
- packed NPU IPC producer。
- HTTP/Ray/callable payload encoder。
- handle 引用生命周期和 post-send sync。

公共层不得直接 import Ray，不得理解 IPC handle tuple，也不得持有 HCCL communicator。

### 8.3 legacy 与 stateful trainer

两种 upstream trainer API 通过 adapter 复用 metadata 和 backend sender：

```text
legacy static trainer_send_weights
  -> legacy adapter
  -> backend sender

stateful TrainerWeightTransferEngine.send_weights
  -> stateful adapter
  -> backend sender
```

adapter 负责调用形态，不改变 wire payload。

## 9. Packed Wire Contract

packed tensor 不是普通内部 helper，而是 producer/consumer 必须一致的跨进程协议。重构前先固定以下字段和语义：

```text
protocol/version（如当前 payload 无该字段，应评估补充）
packed
buffer_size_bytes
num_buffers（适用时）
names
shapes
dtype_names
tensor_sizes
payload/chunk ordering
last-chunk semantics
```

约束：

- producer 与 consumer 使用相同的顺序和边界算法。
- metadata 数组长度必须一致。
- dtype 必须使用稳定字符串表示并进行白名单校验。
- 单 tensor 大于目标 buffer 时必须有确定行为。
- 空 iterator、尾块和多 buffer 轮转必须有 UT。
- HCCL packed payload 与 NPU IPC packed payload 可以使用公共 metadata，但 transport object 不应强行统一。
- wire contract 变化必须被视为协议变更，不能作为普通内部重构提交。

## 10. Device Mapping 与 IPC Identity

### 10.1 Device mapping

`device_mapping.py` 负责：

- 读取 `ASCEND_RT_VISIBLE_DEVICES`。
- logical index 到 physical device id 的映射。
- 对空值、非法整数、重复值和越界进行显式校验。

不得在 import 阶段读取当前 NPU device 或初始化 NPU runtime。

### 10.2 Process identity

`process_identity.py` 负责生成 NPU IPC handle lookup key。当前 `{host_ip}-{physical_device}` 方案必须明确以下约束：

- trainer 与 worker 必须位于允许 IPC 的同一主机/设备域。
- 多网卡、容器网络和 hostname fallback 的选择规则。
- trainer 与 worker 的 visible-device 映射必须可对应到同一 physical id。
- identity 解析失败时在发送/初始化阶段尽早报错。
- cache key 必须包含函数输入；不能让第一次默认设备调用污染后续显式设备查询。

如果 host IP 不能稳定代表 IPC domain，应改用由部署层注入的 node id，而不是继续增强 IP 猜测逻辑。

## 11. HTTP、Ray 与 Callable 边界

### 11.1 公共 HTTP helper

公共 helper 可以提供：

- endpoint URL 拼接。
- timeout。
- `raise_for_status()` 和错误上下文。
- pause/init/start/finish/resume 的基础调用。

公共 helper 不负责：

- HCCL/NPU IPC payload 编码。
- NPU IPC pickle/base64。
- HCCL 阻塞 update 的线程编排。
- backend-specific retry。

### 11.2 安全和错误处理

- 使用 pickle 的 HTTP 路径必须继续受 `VLLM_ALLOW_INSECURE_SERIALIZATION` 控制。
- 不得吞掉 HTTP、Ray 或 callable 抛出的异常。
- 日志应包含 backend、阶段、rank/device 和 send mode，但不得记录完整 IPC handle。
- 自动 cleanup 的行为必须明确：例如 update 失败后是否调用 finish、abort 或仅 resume。
- 默认不自动重试非幂等的 weight update；如需重试，必须有 update/session id 和幂等协议。

## 12. Import 与资源生命周期

必须保持：

- `import vllm_ascend` 不会 import Ray。
- 注册阶段不初始化 NPU、HCCL communicator 或 process group。
- 无 NPU 环境时，可 collection 纯 Python UT。
- `torch_npu`、HCCL wrapper 和 backend client 尽可能在实际使用路径 lazy import。

资源清理要求：

- HCCL 初始化部分失败时回滚已创建资源。
- `shutdown()` 幂等。
- NPU IPC handle 引用在所有接收方完成后释放。
- trainer/client 异常不能让 communicator 或 tensor 引用永久泄漏。
- shutdown 异常需要记录，但应继续尝试清理剩余独立资源。

## 13. 落地顺序

按依赖和风险推荐：

1. 固定支持版本矩阵，增加 `compat.py` 和对应 UT。
2. 收敛 registry/alias，删除分散的 registry 写入。
3. 为现有 packed/unpacked payload 补充 contract UT，不改 transport loop。
4. 抽取 metadata 构造和校验。
5. 将 trainer 逻辑拆成 legacy/stateful adapter 与 backend sender。
6. 抽 HTTP 示例和 e2e helper，保留 backend payload encoder。
7. 抽 device mapping 和 process identity。
8. 在 upstream 行为明确后收敛 lifecycle hook 和错误恢复。
9. 最后评估 packed tensor 的文件级拆分。

每一步必须保持可单独 review 和回滚。不要在同一个提交中同时重写 HCCL transport loop、NPU IPC transport loop 和公共协议。

## 14. 验收标准

### 14.1 第一层：纯 Python UT

无 Ascend 环境必须覆盖：

- legacy/newer API 能力探测。
- native backend 和 alias 注册。
- 重复注册和 registry 恢复。
- import lazy-load 边界。
- lifecycle 合法/非法顺序。
- metadata 长度和 dtype 校验。
- device mapping 正常及非法输入。
- process identity 的 cache 和失败路径。
- HTTP timeout、非 2xx 和异常传播。
- pickle 安全开关。
- packed 空输入、尾块、超大 tensor 和 metadata 不一致。

建议命令：

```text
pytest tests/ut/distributed/weight_transfer
```

当前 tracked 目录主要只有 `test_npu_ipc_engine.py`；上述测试需要随对应模块逐步新增，不能将历史 `.pyc` 视为覆盖。

### 14.2 第二层：mock transport

- mock HCCL communicator，验证 broadcast 顺序和 cleanup。
- mock NPU IPC reduce/rebuild，验证 UUID lookup 和 handle 生命周期。
- HTTP、Ray、callable 三种 send mode 的 payload 与异常传播。
- legacy static 与 stateful trainer 产生等价 wire payload。
- update 中途失败后的状态和 cleanup。

### 14.3 第三层：NPU e2e

```text
pytest tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py
pytest tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py
```

必须覆盖：

- HCCL 完整 update。
- NPU IPC 完整 update。
- pause/resume。
- init/start/update/finish 顺序。
- packed=False 和 packed=True。
- HTTP send mode。
- 已支持的 Ray/callable 模式。
- 错误输入返回明确错误。

### 14.4 行为和兼容性

重构前后必须保持：

- backend name 和 alias 兼容。
- init/update payload 兼容。
- trainer 与 worker 的 packed 配置一致性校验。
- 日志可以定位 backend、阶段、rank、device 和 send mode。
- 不提前 import Ray 或初始化 NPU/HCCL。
- 非法状态、非法 metadata 和 identity mismatch 不会静默成功。
- `shutdown()` 与失败清理可重复执行。

## 15. 非目标

本轮重构不做：

- 重新设计 upstream HTTP API。
- 将 HCCL 和 NPU IPC 合并为统一 transport。
- 引入新的序列化格式替换现有 wire contract。
- 优化通信性能或 buffer 大小。
- 新增跨主机 NPU IPC 能力。
- 顺带重构模型加载、pause/resume 或其他非 weight transfer 模块。

## 16. 最终判断

本次重构的价值在于集中 upstream API 同步点，同时保留 Ascend backend 的独立错误模型和资源生命周期。

理想结果是：

```text
upstream API 变化
  -> compat / registry / adapter

公共 metadata 或 lifecycle 变化
  -> metadata / lifecycle / HTTP helper

Ascend transport 变化
  -> hccl 或 npu_ipc backend/sender

packed wire contract 变化
  -> 显式协议变更和 producer/consumer 联合测试
```

如果重构后公共模块仍直接操作 HCCL communicator、IPC handle、Ray actor 或 `torch.npu`，说明边界过宽；如果版本判断、HTTP lifecycle 和 metadata 构造仍复制在多个 backend 和示例中，说明同步点仍未真正收敛。
