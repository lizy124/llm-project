# vllm-ascend weight_transfer 重构记录

## 需要先明确的问题

当前 vllm-ascend 的 `weight_transfer` 不是一套完全独立设计的系统，而是把 upstream vLLM 的 `nccl` / `ipc` weight transfer 能力 port 到 Ascend NPU 场景：

```text
upstream nccl -> Ascend hccl
upstream ipc  -> Ascend npu_ipc
```

这种 porting 方式短期能快速落地，但现在已经形成一个问题：重复边界不清。

主要表现是，大量 upstream 侧的公共流程被复制到了 Ascend backend、示例和测试里，包括：

```text
weight transfer lifecycle:
  init_weight_transfer_engine
  start_weight_update
  update_weights
  finish_weight_update

trainer-side send orchestration:
  参数遍历
  metadata 收集
  packed / unpacked 分支
  send_mode 分发
  post-send sync

HTTP 示例流程:
  pause
  get_world_size
  init
  start
  update
  finish
  resume
```

这些逻辑本身不是 HCCL 或 NPU IPC 独有能力，而是 upstream vLLM weight transfer 的公共协议和调用编排。它们被复制到多个 Ascend 文件后，会导致后续 upstream API 对齐成本持续增加。

例如 upstream 后续调整 `start_weight_update` / `finish_weight_update` 语义、trainer send payload、HTTP endpoint request schema 或 packed tensor 协议时，vllm-ascend 需要在 HCCL backend、NPU IPC backend、示例、e2e 测试中多处同步修改。修改点越分散，越容易出现某个 backend 或某个示例漏改，最终表现为接口能 import 但实际 weight update 流程不一致。

因此，这次重构要解决的核心问题不是简单减少代码行数，而是重新划清边界：

```text
upstream 公共协议和流程
  -> 收敛到公共 helper / lifecycle / trainer_send / registry 层

Ascend 特有通信差异
  -> 保留在 hccl_engine / npu_ipc_engine / packed transport 中

示例和测试中的重复调用模板
  -> 收敛到示例 helper 和测试 helper
```

重构完成后，代码应该能让 reviewer 很快判断：哪些地方是跟 upstream weight transfer API 对齐的公共逻辑，哪些地方是真正的 Ascend HCCL / NPU IPC 差异。

## 重构方案

### 1. 收敛 backend 注册和 alias 逻辑

当前注册入口分散在：

```text
vllm_ascend/distributed/weight_transfer/__init__.py
vllm_ascend/patch/platform/patch_weight_transfer_engine.py
```

建议新增统一注册入口：

```text
vllm_ascend/distributed/weight_transfer/registry.py
```

提供一个明确函数：

```python
def register_ascend_weight_transfer_engines(
    include_upstream_aliases: bool = True,
    override_existing: bool = True,
) -> None:
    ...
```

集中维护以下关系：

```text
hccl    -> HCCLWeightTransferEngine
npu_ipc -> NPUIPCWeightTransferEngine
nccl    -> HCCLWeightTransferEngine       # Ascend compatibility alias
ipc     -> NPUIPCWeightTransferEngine     # Ascend compatibility alias
```

`__init__.py` 只负责暴露注册函数，`patch_weight_transfer_engine.py` 只负责调用注册函数。

这样做解决的问题：

- backend name 和 upstream-compatible alias 不再分散维护。
- 后续 upstream factory 注册 API 或 backend 类型规则变化时，优先只改 registry。
- 减少直接散落访问 `WeightTransferEngineFactory._registry` 的位置。

### 2. 抽出 HTTP lifecycle 示例 helper

当前 `rlhf_http_hccl.py` / `rlhf_http_npu_ipc.py` 和 upstream 示例高度相似，重复了大量 HTTP endpoint 调用模板。

建议新增：

```text
examples/rl/weight_transfer_http_utils.py
```

封装公共 HTTP 操作：

```python
def pause_generation(url): ...
def resume_generation(url): ...
def get_world_size(url): ...
def init_weight_transfer_engine(url, init_info): ...
def start_weight_update(url): ...
def update_weights(url, update_info): ...
def finish_weight_update(url): ...
def run_weight_update_lifecycle(...): ...
```

HCCL / NPU IPC 示例只保留真正不同的内容：

```text
backend config
init_info / update_info 构造
trainer_send_weights 调用参数
HCCL 或 NPU IPC 的设备/通信差异
```

这样做解决的问题：

- HTTP endpoint schema 或调用顺序变化时，示例只需要改 helper。
- 示例文件不再复制完整 lifecycle，更容易看出 backend 差异。
- e2e 测试也可以复用同一套 HTTP 调用模板。

### 3. 抽出测试 helper

当前 e2e 测试容易重复服务启动、endpoint 调用、权重变更断言等流程。

建议新增公共测试工具，例如：

```text
tests/e2e/pull_request/weight_transfer_utils.py
```

封装：

```text
启动和等待 vLLM server
构造 --weight-transfer-config
调用 pause / init / start / update / finish / resume
检查权重更新前后输出变化
检查异常路径和接口返回
```

backend 测试只保留场景差异：

```text
one-card npu_ipc
two-card hccl
packed / unpacked
Ray / HTTP / callable send mode
```

这样做解决的问题：

- 测试不再通过复制流程来覆盖 backend。
- upstream API 变化时，测试入口集中修改。
- 后续新增 sparse_hccl 或 stateful trainer client 时，不需要重新复制一套 e2e 调用链。

### 4. 抽出 lifecycle policy

HCCL 和 NPU IPC 都要实现 `start_weight_update` / `finish_weight_update`，但行为不同：

```text
HCCL: 需要配合 layerwise reload 做初始化和收尾
NPU IPC: 当前基本是 no-op
```

建议新增：

```text
vllm_ascend/distributed/weight_transfer/lifecycle.py
```

用轻量 policy 表达差异：

```python
class WeightUpdateLifecyclePolicy:
    def start(self, model): ...
    def finish(self, model, model_config): ...

class LayerwiseReloadLifecyclePolicy(WeightUpdateLifecyclePolicy): ...
class NoopLifecyclePolicy(WeightUpdateLifecyclePolicy): ...
```

backend 中只选择 policy，不直接散落 lifecycle 细节。

这样做解决的问题：

- upstream 调整 start/finish 生命周期语义时，优先改 lifecycle 层。
- HCCL 和 NPU IPC 的生命周期差异被显式表达。
- backend 文件不再同时承担通信实现和生命周期模板职责。

### 5. 抽出 device mapping

NPU IPC 中有 Ascend 特有的 device / host / physical chip id 逻辑，例如：

```text
ASCEND_RT_VISIBLE_DEVICES
logical device index
physical chip id
host ip
NPU IPC uuid
```

建议新增：

```text
vllm_ascend/distributed/weight_transfer/device_mapping.py
```

集中放置：

```python
def get_host_ip(): ...
def resolve_physical_device_id(...): ...
def make_npu_ipc_device_key(...): ...
```

这样做解决的问题：

- NPU IPC backend 中的 handle 编解码逻辑更清楚。
- device 映射可以用纯 Python UT 覆盖，不依赖完整 e2e。
- 后续排查 `ASCEND_RT_VISIBLE_DEVICES` 相关问题时入口明确。

### 6. 抽出 trainer_send orchestration

trainer-side send 是 upstream API drift 的高风险区。当前 HCCL / NPU IPC 都包含参数遍历、metadata 收集、packed/unpacked 分支、send_mode 分发、同步等编排逻辑。

建议新增：

```text
vllm_ascend/distributed/weight_transfer/trainer_send.py
```

公共层负责：

```text
遍历待发送参数
收集 name / shape / dtype metadata
处理 rank0-only send
处理 HTTP / Ray / callable send_mode
处理 post-send sync
构造公共 update payload
```

backend 只提供 transport-specific hook：

```text
HCCL:
  初始化 HCCL process group
  broadcast tensor / packed buffer

NPU IPC:
  生成 NPU IPC handle
  rebuild NPU tensor
  packed NPU IPC producer / consumer
```

这样做解决的问题：

- trainer payload 或 send_mode 变化时，公共编排集中修改。
- HCCL / NPU IPC backend 更薄，差异集中在通信层。
- 后续 stateful trainer client 接入时，不需要从多个 backend 文件里拆流程。

### 7. 最后再拆 packed tensor

`packed_tensor.py` 同时承载 HCCL packed broadcast 和 NPU IPC packed handle 逻辑，建议最后处理，因为这里最贴近实际传输路径，风险比 registry/helper 更高。

目标结构可以是：

```text
vllm_ascend/distributed/weight_transfer/packed/
  __init__.py
  common.py
  hccl.py
  npu_ipc.py
```

职责划分：

```text
common.py: chunk planning、metadata、buffer size 校验
hccl.py: HCCL packed broadcast producer / consumer
npu_ipc.py: NPU IPC packed producer / consumer
```

旧的 `packed_tensor.py` 可以先保留为兼容 re-export，降低一次性迁移风险。

这样做解决的问题：

- packed common 和 transport-specific 实现分离。
- 后续 upstream packed 协议变化时，公共 metadata / chunk 逻辑有集中入口。
- 避免一个文件同时混合两种传输模型。

## 推荐落地顺序

不建议一次性大改。推荐按风险从低到高分阶段推进：

```text
第一轮：registry + alias 收敛
第二轮：HTTP 示例 helper + 测试 helper
第三轮：lifecycle policy + device mapping
第四轮：trainer_send orchestration
第五轮：packed tensor 拆分
```

第一轮和第二轮主要移动调用模板，不改核心传输行为，适合先落地。第三轮开始影响 backend 内部结构，需要 UT 兜底。第四轮和第五轮触碰 trainer send 和 packed 传输路径，必须在已有 UT/e2e 能覆盖后再做。

## 重构验收标准

重构是否成功，不能只看“代码变少了”或“文件拆开了”。需要从行为、边界、可维护性和回归风险四个方面验收。

### 1. 行为等价验收

重构前后，已有功能必须保持一致。

至少需要覆盖：

```text
HCCL backend 可以完成一次完整 weight update
NPU IPC backend 可以完成一次完整 weight update
pause / resume 正常工作
init / start / update / finish 调用顺序正常
packed=False 路径正常
packed=True 路径正常
HTTP send mode 正常
Ray 或 callable send mode 如果已有覆盖，保持正常
错误输入仍返回预期错误，而不是静默成功
```

推荐验证命令：

```bash
pytest tests/ut/distributed/weight_transfer
pytest tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py
pytest tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py
```

如果本地没有 Ascend NPU 环境，最低验收要求是：

```text
纯 Python UT 通过
import collection 通过
registry / lifecycle / device_mapping / trainer_send helper 的 mock 测试通过
NPU 相关 e2e 留给 CI，并在 PR 中明确说明本地未跑原因
```

### 2. 接口兼容验收

重构后必须继续支持两类 backend 名：

```text
Ascend 原生 backend:
  hccl
  npu_ipc

upstream-compatible alias:
  nccl
  ipc
```

验收方式：

```text
--weight-transfer-config '{"backend": "hccl"}' 能创建 HCCLWeightTransferEngine
--weight-transfer-config '{"backend": "npu_ipc"}' 能创建 NPUIPCWeightTransferEngine
--weight-transfer-config '{"backend": "nccl"}' 在 Ascend 环境下映射到 HCCLWeightTransferEngine
--weight-transfer-config '{"backend": "ipc"}' 在 Ascend 环境下映射到 NPUIPCWeightTransferEngine
```

这部分应该有 registry UT，不应只依赖 e2e。

### 3. 边界清晰验收

重构后 reviewer 应该能按职责定位代码：

```text
backend 注册和 alias -> registry.py
start / finish 生命周期差异 -> lifecycle.py
HTTP 示例调用模板 -> weight_transfer_http_utils.py
测试服务和 endpoint 模板 -> weight_transfer_utils.py
device / physical chip id / host key -> device_mapping.py
trainer send 编排 -> trainer_send.py
HCCL 通信细节 -> hccl_engine.py
NPU IPC handle 编解码 -> npu_ipc_engine.py
packed 公共和 transport 差异 -> packed/
```

如果重构后公共 lifecycle、HTTP endpoint 调用、trainer send payload 仍然在多个 backend 或示例中复制，说明问题没有真正解决，只是移动了代码。

### 4. upstream 对齐成本验收

可以用一次模拟 upstream API 变化来验收重构效果。

例如假设 upstream 修改了：

```text
start_weight_update 增加 request 字段
finish_weight_update 返回状态
update_weights payload 增加 session_id
HTTP endpoint helper 需要统一处理错误码
```

重构后的理想修改点应该集中在：

```text
lifecycle.py
trainer_send.py
weight_transfer_http_utils.py
tests/e2e/pull_request/weight_transfer_utils.py
```

不应该需要同时修改：

```text
hccl_engine.py
npu_ipc_engine.py
rlhf_http_hccl.py
rlhf_http_npu_ipc.py
多个 e2e 测试文件
```

如果一次 upstream API 小变化仍然需要多处 backend 和示例同步，说明重构没有达到目标。

### 5. 回归风险验收

重构必须避免制造新问题，尤其是 import 副作用和设备依赖问题。

需要确认：

```text
import vllm_ascend 不会提前 import Ray 或初始化 NPU 通信资源
registry lazy-load 行为保持不变
无 NPU 环境下，纯 Python 测试不会因为 import torch_npu / HCCL 失败而整体失败
错误路径不会吞异常
日志仍能定位 backend、rank、device、send mode
```

这部分可以通过 import collection 测试、mock 测试和 CI job 验证。

### 6. 文档和示例验收

重构完成后，示例应该更短，但不能丢失可运行性。

需要确认：

```text
rlhf_http_hccl.py 仍然能作为 HCCL 用户入口阅读和运行
rlhf_http_npu_ipc.py 仍然能作为 NPU IPC 用户入口阅读和运行
公共 helper 名称清楚，不要求用户理解内部 backend 实现
示例中 backend config、启动命令、pause/init/start/update/finish/resume 顺序仍然明确
```

示例不是为了展示所有 helper 细节，而是为了让用户知道该怎么跑 HCCL 或 NPU IPC weight transfer。

## 可以认为重构没问题的条件

只有同时满足下面条件，才可以认为这次重构是解决问题，而不是制造问题：

```text
1. 现有 HCCL / NPU IPC weight update 行为不变，UT/e2e 或 CI 通过。
2. hccl/npu_ipc/nccl/ipc 四类 backend name 兼容关系明确，并有测试覆盖。
3. HTTP lifecycle、trainer send、lifecycle start/finish、device mapping 不再散落复制。
4. backend 文件主要保留通信差异，而不是继续承载大量 upstream 公共流程。
5. 示例和测试复用 helper，但用户仍能清楚看到每个 backend 怎么运行。
6. import lazy-load 和无设备环境下的测试收集不回退。
7. 对一次模拟 upstream API 变化，修改点明显比重构前更集中。
```

如果只完成了文件拆分，但公共流程仍然复制在多个地方，或者 e2e 需要大量特殊分支才能跑通，就不能算完成重构目标。
